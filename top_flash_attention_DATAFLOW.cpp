#include "dcl_optimized.h"
#include "hls_stream.h"

// Stream을 통해 전달할 KV 블록 데이터 구조체
struct KV_Block {
    qint8_t K[Bc][dk];
    qint8_t V[Bc][dv];
    scale_fixed_t scale_K[Bc];
    scale_fixed_t scale_V[Bc];
};

// Load KV 함수 - 메모리에서 K, V 읽어서 stream으로 출력
void load_kv_task(
    qint8_t K[N][dk],
    qint8_t V[N][dv],
    float scale_K[N],
    float scale_V[N],
    int j,
    hls::stream<KV_Block>& kv_stream
) {
    #pragma HLS INLINE off

    KV_Block block;

    LOAD_KV:
    for (int c = 0; c < Bc; c++) {
        #pragma HLS PIPELINE II=1
        block.scale_K[c] = (scale_fixed_t)scale_K[j + c];
        block.scale_V[c] = (scale_fixed_t)scale_V[j + c];
        for (int k = 0; k < dk; k++) {
            block.K[c][k] = K[j + c][k];
        }
        for (int v = 0; v < dv; v++) {
            block.V[c][v] = V[j + c][v];
        }
    }

    kv_stream.write(block);
}

// Process 함수 - stream에서 KV 블록 받아서 attention 계산
void process_task(
    hls::stream<KV_Block>& kv_stream,
    qint8_t local_Q[Br][dk],
    scale_fixed_t local_scale_Q[Br],
    ap_fixed<32,16> local_O[Br][dv],
    ap_fixed<32,16> local_m[Br],
    ap_fixed<32,16> local_l[Br]
) {
    #pragma HLS INLINE off

    KV_Block block = kv_stream.read();

    // 로컬 복사 (stream에서 읽은 데이터)
    qint8_t local_K[Bc][dk];
    #pragma HLS ARRAY_PARTITION variable=local_K cyclic factor=4 dim=2
    qint8_t local_V[Bc][dv];
    #pragma HLS ARRAY_PARTITION variable=local_V cyclic factor=4 dim=2
    scale_fixed_t local_scale_K[Bc];
    scale_fixed_t local_scale_V[Bc];

    // 복사
    for (int c = 0; c < Bc; c++) {
        #pragma HLS PIPELINE II=1
        local_scale_K[c] = block.scale_K[c];
        local_scale_V[c] = block.scale_V[c];
        for (int k = 0; k < dk; k++) {
            local_K[c][k] = block.K[c][k];
        }
        for (int v = 0; v < dv; v++) {
            local_V[c][v] = block.V[c][v];
        }
    }

    PROCESS_ROW:
    for (int r = 0; r < Br; r++) {

        ap_fixed<32,16> scores[Bc];
        #pragma HLS ARRAY_PARTITION variable=scores complete
        ap_fixed<32,16> row_max_val = -10000.0;

        SCORE_LOOP:
        for (int c = 0; c < Bc; c++) {
            #pragma HLS PIPELINE II=1
            qint32_t score_sum_int = 0;

            SCORE_DOT:
            for (int k = 0; k < dk; k++) {
                #pragma HLS UNROLL factor=4
                score_sum_int += local_Q[r][k] * local_K[c][k];
            }

            auto combined_scale = local_scale_Q[r] * local_scale_K[c];
            auto raw_score = score_sum_int * combined_scale;
            scores[c] = (ap_fixed<32, 16>)(raw_score >> 3);

            if (scores[c] > row_max_val) {
                row_max_val = scores[c];
            }
        }

        ap_fixed<32,16> m_prev = local_m[r];
        ap_fixed<32,16> m_new = (m_prev > row_max_val) ? m_prev : row_max_val;
        ap_fixed<32,16> correction_prev = hls::exp((float)(m_prev - m_new));

        ap_fixed<32,16> p_sum_curr = 0;
        ap_fixed<32,16> P[Bc];
        #pragma HLS ARRAY_PARTITION variable=P complete

        SOFTMAX_LOOP:
        for (int c = 0; c < Bc; c++) {
            #pragma HLS PIPELINE II=1
            P[c] = hls::exp((float)(scores[c] - m_new));
            p_sum_curr += P[c];
        }

        ap_fixed<32, 16> scaled_P[Bc];
        #pragma HLS ARRAY_PARTITION variable=scaled_P complete

        PRE_SCALE_LOOP:
        for (int c = 0; c < Bc; c++) {
            #pragma HLS PIPELINE II=1
            scaled_P[c] = P[c] * local_scale_V[c];
        }

        local_l[r] = local_l[r] * correction_prev + p_sum_curr;
        local_m[r] = m_new;

        OUTPUT_UPDATE:
        for (int v = 0; v < dv; v++) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32,16> weighted_sum = 0;

            WEIGHTED_SUM:
            for (int c = 0; c < Bc; c++) {
                #pragma HLS UNROLL factor=4
                auto term = scaled_P[c] * local_V[c][v];
                weighted_sum += term;
            }
            local_O[r][v] = local_O[r][v] * correction_prev + weighted_sum;
        }
    }
}


void compute_attention_HLS(
    qint8_t Q[N][dk],
    qint8_t K[N][dk],
    qint8_t V[N][dv],
    fixed_t Output[N][dv],
    float scale_Q[N],
    float scale_K[N],
    float scale_V[N]
) {
    //bus[0]
    #pragma HLS INTERFACE mode=m_axi port=Q       bundle=gmem0 depth=N*dk
    #pragma HLS INTERFACE mode=m_axi port=scale_Q bundle=gmem0 depth=N

    //bus[1]
    #pragma HLS INTERFACE mode=m_axi port=K       bundle=gmem1 depth=N*dk
    #pragma HLS INTERFACE mode=m_axi port=scale_K bundle=gmem1 depth=N

    //bus[2]
    #pragma HLS INTERFACE mode=m_axi port=V       bundle=gmem2 depth=N*dv
    #pragma HLS INTERFACE mode=m_axi port=scale_V bundle=gmem2 depth=N

    //bus[3]
    #pragma HLS INTERFACE mode=m_axi port=Output  bundle=gmem3 depth=N*dv

    #pragma HLS INTERFACE mode=s_axilite port=return

    // Local buffers for Q
    qint8_t local_Q[Br][dk];
    #pragma HLS ARRAY_PARTITION variable=local_Q cyclic factor=4 dim=2

    scale_fixed_t local_scale_Q[Br];

    // Output accumulators
    ap_fixed<32,16> local_O[Br][dv];
    ap_fixed<32,16> local_m[Br];
    #pragma HLS ARRAY_PARTITION variable=local_m complete
    ap_fixed<32,16> local_l[Br];
    #pragma HLS ARRAY_PARTITION variable=local_l complete

    const int num_kv_blocks = N / Bc;

    OUTER_Q_LOOP:
    for (int i = 0; i < N; i += Br) {

        // Load Q block and scales
        LOAD_Q:
        for (int r = 0; r < Br; r++) {
            #pragma HLS PIPELINE II=1
            local_scale_Q[r] = (scale_fixed_t)scale_Q[i + r];
            for (int k = 0; k < dk; k++) {
                local_Q[r][k] = Q[i + r][k];
            }
        }

        // Initialize
        INIT_STATS:
        for (int r = 0; r < Br; r++) {
            #pragma HLS UNROLL
            local_m[r] = -10000.0;
            local_l[r] = 0;
            for (int c = 0; c < dv; c++) {
                #pragma HLS PIPELINE II=1
                local_O[r][c] = 0;
            }
        }

        // Stream for KV blocks
        hls::stream<KV_Block> kv_stream;
        #pragma HLS STREAM variable=kv_stream depth=2

        OUTER_KV_LOOP:
        for (int jb = 0; jb < num_kv_blocks; jb++) {
            #pragma HLS DATAFLOW

            int j = jb * Bc;

            // Task 1: Load KV block
            load_kv_task(K, V, scale_K, scale_V, j, kv_stream);

            // Task 2: Process attention
            process_task(kv_stream, local_Q, local_scale_Q, local_O, local_m, local_l);
        }

        WRITE_OUTPUT:
        for (int r = 0; r < Br; r++) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32,16> inv_sum = ap_fixed<32, 16>(1.0) / local_l[r];
            for (int v = 0; v < dv; v++) {
                Output[i + r][v] = (fixed_t)(local_O[r][v] * inv_sum);
            }
        }
    } // end OUTER_Q_LOOP
}