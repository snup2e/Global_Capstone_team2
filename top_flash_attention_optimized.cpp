#include "dcl_optimized.h"
#include "hls_stream.h"

// --------------------------------------------------------
// 최적화 설정
// - 타일 크기: Br=16, Bc=16 (BRAM 감소)
// - 데이터: INT8 + per-row scale factor
// - exp(): hls::exp() 사용
// --------------------------------------------------------

#define Br 16
#define Bc 16
#define FIFO_DEPTH 4

// --------------------------------------------------------
// Stream 타입 정의 (비트폭 최소화)
// --------------------------------------------------------
typedef struct {
    int8_t data[dk];
    float scale;
} q_row_t;  // dk*8 + 32 = 544 bits

typedef struct {
    int8_t data[dk];
    float scale;
} k_row_t;  // dk*8 + 32 = 544 bits

typedef struct {
    int8_t data[dv];
    float scale;
} v_row_t;  // dv*8 + 32 = 544 bits

typedef struct {
    calc_t scores[Bc];
    calc_t row_max;
    int row_idx;
} score_packet_t;

typedef struct {
    calc_t P[Bc];
    calc_t m_new;
    calc_t correction;
    calc_t p_sum;
    int row_idx;
} softmax_packet_t;

// --------------------------------------------------------
// Stage 1: Q 블록 로드 (INT8 + scale)
// --------------------------------------------------------
void stage_load_Q(
    int8_t* Q,
    float* Q_scale,
    hls::stream<q_row_t> &Q_stream,
    int i_offset
) {
    #pragma HLS INLINE off
    
    LOAD_Q_ROWS:
    for (int r = 0; r < Br; r++) {
        q_row_t q_row;
        q_row.scale = Q_scale[i_offset + r];
        
        LOAD_Q_COLS:
        for (int k = 0; k < dk; k++) {
            #pragma HLS PIPELINE II=1
            q_row.data[k] = Q[(i_offset + r) * dk + k];
        }
        Q_stream.write(q_row);
    }
}

// --------------------------------------------------------
// Stage 2: K 블록 로드 (INT8 + scale, 행 단위)
// --------------------------------------------------------
void stage_load_K(
    int8_t* K,
    float* K_scale,
    hls::stream<k_row_t> &K_stream,
    int num_blocks
) {
    #pragma HLS INLINE off
    
    LOAD_K_BLOCKS:
    for (int j = 0; j < num_blocks; j++) {
        LOAD_K_ROWS:
        for (int c = 0; c < Bc; c++) {
            k_row_t k_row;
            int row_idx = j * Bc + c;
            k_row.scale = K_scale[row_idx];
            
            LOAD_K_COLS:
            for (int k = 0; k < dk; k++) {
                #pragma HLS PIPELINE II=1
                k_row.data[k] = K[row_idx * dk + k];
            }
            K_stream.write(k_row);
        }
    }
}

// --------------------------------------------------------
// Stage 3: V 블록 로드 (INT8 + scale, 행 단위)
// --------------------------------------------------------
void stage_load_V(
    int8_t* V,
    float* V_scale,
    hls::stream<v_row_t> &V_stream,
    int num_blocks
) {
    #pragma HLS INLINE off
    
    LOAD_V_BLOCKS:
    for (int j = 0; j < num_blocks; j++) {
        LOAD_V_ROWS:
        for (int c = 0; c < Bc; c++) {
            v_row_t v_row;
            int row_idx = j * Bc + c;
            v_row.scale = V_scale[row_idx];
            
            LOAD_V_COLS:
            for (int v = 0; v < dv; v++) {
                #pragma HLS PIPELINE II=1
                v_row.data[v] = V[row_idx * dv + v];
            }
            V_stream.write(v_row);
        }
    }
}

// --------------------------------------------------------
// Stage 4: Score 계산 (Q * K^T)
// INT8 dot product + scale factor 적용
// --------------------------------------------------------
void stage_compute_scores(
    hls::stream<q_row_t> &Q_stream,
    hls::stream<k_row_t> &K_stream,
    hls::stream<score_packet_t> &score_stream,
    int num_kv_blocks
) {
    #pragma HLS INLINE off
    
    // Q 버퍼 (INT8 + scale)
    int8_t Q_data[Br][dk];
    #pragma HLS ARRAY_PARTITION variable=Q_data cyclic factor=8 dim=2
    float Q_scales[Br];
    #pragma HLS ARRAY_PARTITION variable=Q_scales complete
    
    // K 버퍼 (INT8 + scale) - 한 블록씩
    int8_t K_data[Bc][dk];
    #pragma HLS ARRAY_PARTITION variable=K_data cyclic factor=8 dim=2
    float K_scales[Bc];
    #pragma HLS ARRAY_PARTITION variable=K_scales complete
    
    // Q 읽기
    LOAD_Q:
    for (int r = 0; r < Br; r++) {
        q_row_t q_row = Q_stream.read();
        Q_scales[r] = q_row.scale;
        COPY_Q:
        for (int k = 0; k < dk; k++) {
            #pragma HLS UNROLL factor=8
            Q_data[r][k] = q_row.data[k];
        }
    }
    
    // 1/sqrt(dk)
    const calc_t inv_sqrt_dk = 0.125;  // 1/sqrt(64) = 0.125
    
    // 각 K 블록에 대해
    KV_BLOCKS:
    for (int j = 0; j < num_kv_blocks; j++) {
        // K 블록 로드
        RECV_K_ROWS:
        for (int c = 0; c < Bc; c++) {
            k_row_t k_row = K_stream.read();
            K_scales[c] = k_row.scale;
            COPY_K:
            for (int k = 0; k < dk; k++) {
                #pragma HLS UNROLL factor=8
                K_data[c][k] = k_row.data[k];
            }
        }
        
        // Score 계산
        Q_ROWS:
        for (int r = 0; r < Br; r++) {
            score_packet_t pkt;
            pkt.row_max = calc_t(-10000.0);
            pkt.row_idx = r;
            
            calc_t q_scale = (calc_t)Q_scales[r];
            
            COMPUTE_SCORES:
            for (int c = 0; c < Bc; c++) {
                #pragma HLS PIPELINE II=1
                
                // INT8 dot product
                ap_int<32> dot_sum = 0;
                
                DOT_PRODUCT:
                for (int k = 0; k < dk; k++) {
                    #pragma HLS UNROLL factor=8
                    dot_sum += (ap_int<16>)Q_data[r][k] * (ap_int<16>)K_data[c][k];
                }
                
                // Scale 적용: score = dot_sum * Q_scale * K_scale * (1/sqrt(dk))
                calc_t k_scale = (calc_t)K_scales[c];
                pkt.scores[c] = (calc_t)dot_sum * q_scale * k_scale * inv_sqrt_dk;
                
                if (pkt.scores[c] > pkt.row_max) {
                    pkt.row_max = pkt.scores[c];
                }
            }
            
            score_stream.write(pkt);
        }
    }
}

// --------------------------------------------------------
// Stage 5: Softmax 업데이트 (Online Softmax)
// --------------------------------------------------------
void stage_softmax_update(
    hls::stream<score_packet_t> &score_stream,
    hls::stream<softmax_packet_t> &softmax_stream,
    calc_t local_m[Br],
    calc_t local_l[Br],
    int num_kv_blocks
) {
    #pragma HLS INLINE off
    
    // 초기화
    INIT_STATS:
    for (int r = 0; r < Br; r++) {
        #pragma HLS UNROLL
        local_m[r] = calc_t(-10000.0);
        local_l[r] = calc_t(0.0);
    }
    
    SOFTMAX_BLOCKS:
    for (int j = 0; j < num_kv_blocks; j++) {
        SOFTMAX_ROWS:
        for (int r = 0; r < Br; r++) {
            score_packet_t score_pkt = score_stream.read();
            softmax_packet_t sm_pkt;
            
            int idx = score_pkt.row_idx;
            sm_pkt.row_idx = idx;
            
            // Online softmax
            calc_t m_prev = local_m[idx];
            sm_pkt.m_new = (m_prev > score_pkt.row_max) ? m_prev : score_pkt.row_max;
            sm_pkt.correction = hls::exp((float)(m_prev - sm_pkt.m_new));
            
            sm_pkt.p_sum = calc_t(0.0);
            
            EXP_LOOP:
            for (int c = 0; c < Bc; c++) {
                #pragma HLS PIPELINE II=1
                sm_pkt.P[c] = hls::exp((float)(score_pkt.scores[c] - sm_pkt.m_new));
                sm_pkt.p_sum += sm_pkt.P[c];
            }
            
            local_l[idx] = local_l[idx] * sm_pkt.correction + sm_pkt.p_sum;
            local_m[idx] = sm_pkt.m_new;
            
            softmax_stream.write(sm_pkt);
        }
    }
}

// --------------------------------------------------------
// Stage 6: Output 누적 (P * V)
// --------------------------------------------------------
void stage_accumulate_output(
    hls::stream<softmax_packet_t> &softmax_stream,
    hls::stream<v_row_t> &V_stream,
    calc_t local_O[Br][dv],
    int num_kv_blocks
) {
    #pragma HLS INLINE off
    
    // V 버퍼 (INT8 + scale)
    int8_t V_data[Bc][dv];
    #pragma HLS ARRAY_PARTITION variable=V_data cyclic factor=8 dim=2
    float V_scales[Bc];
    #pragma HLS ARRAY_PARTITION variable=V_scales complete
    
    // O 초기화
    INIT_O:
    for (int r = 0; r < Br; r++) {
        for (int v = 0; v < dv; v++) {
            #pragma HLS UNROLL factor=8
            local_O[r][v] = calc_t(0.0);
        }
    }
    
    ACC_BLOCKS:
    for (int j = 0; j < num_kv_blocks; j++) {
        // V 블록 로드
        RECV_V_ROWS:
        for (int c = 0; c < Bc; c++) {
            v_row_t v_row = V_stream.read();
            V_scales[c] = v_row.scale;
            COPY_V:
            for (int v = 0; v < dv; v++) {
                #pragma HLS UNROLL factor=8
                V_data[c][v] = v_row.data[v];
            }
        }
        
        ACC_ROWS:
        for (int r = 0; r < Br; r++) {
            softmax_packet_t sm_pkt = softmax_stream.read();
            int idx = sm_pkt.row_idx;
            
            ACC_DIMS:
            for (int v = 0; v < dv; v++) {
                #pragma HLS PIPELINE II=1
                
                calc_t weighted_sum = calc_t(0.0);
                
                MAC_LOOP:
                for (int c = 0; c < Bc; c++) {
                    #pragma HLS UNROLL factor=4
                    // Dequantize V and multiply with P
                    calc_t v_val = (calc_t)V_data[c][v] * (calc_t)V_scales[c];
                    weighted_sum += sm_pkt.P[c] * v_val;
                }
                
                local_O[idx][v] = local_O[idx][v] * sm_pkt.correction + weighted_sum;
            }
        }
    }
}

// --------------------------------------------------------
// Stage 7: 최종 정규화 및 저장
// --------------------------------------------------------
void stage_normalize_store(
    fixed_t* Output,
    calc_t local_O[Br][dv],
    calc_t local_l[Br],
    int i_offset
) {
    #pragma HLS INLINE off
    
    STORE_ROWS:
    for (int r = 0; r < Br; r++) {
        calc_t inv_sum = calc_t(1.0) / local_l[r];
        
        STORE_COLS:
        for (int v = 0; v < dv; v++) {
            #pragma HLS PIPELINE II=1
            Output[(i_offset + r) * dv + v] = (fixed_t)(local_O[r][v] * inv_sum);
        }
    }
}

// --------------------------------------------------------
// TOP FUNCTION
// --------------------------------------------------------
void compute_attention_HLS(
    int8_t Q[N][dk],
    int8_t K[N][dk],
    int8_t V[N][dv],
    float Q_scale[N],
    float K_scale[N],
    float V_scale[N],
    fixed_t Output[N][dv]
) {
    #pragma HLS INTERFACE mode=m_axi port=Q bundle=gmem0 depth=N*dk
    #pragma HLS INTERFACE mode=m_axi port=K bundle=gmem1 depth=N*dk
    #pragma HLS INTERFACE mode=m_axi port=V bundle=gmem2 depth=N*dv
    #pragma HLS INTERFACE mode=m_axi port=Q_scale bundle=gmem3 depth=N
    #pragma HLS INTERFACE mode=m_axi port=K_scale bundle=gmem4 depth=N
    #pragma HLS INTERFACE mode=m_axi port=V_scale bundle=gmem5 depth=N
    #pragma HLS INTERFACE mode=m_axi port=Output bundle=gmem6 depth=N*dv
    
    const int num_kv_blocks = N / Bc;
    
    // Q 블록별 처리
    OUTER_Q_LOOP:
    for (int i = 0; i < N; i += Br) {
        // Statistics 버퍼
        calc_t local_m[Br];
        #pragma HLS ARRAY_PARTITION variable=local_m complete
        
        calc_t local_l[Br];
        #pragma HLS ARRAY_PARTITION variable=local_l complete
        
        calc_t local_O[Br][dv];
        #pragma HLS ARRAY_PARTITION variable=local_O cyclic factor=8 dim=2
        
        // FIFO 스트림
        hls::stream<q_row_t> Q_stream("Q_stream");
        #pragma HLS STREAM variable=Q_stream depth=FIFO_DEPTH
        
        hls::stream<k_row_t> K_stream("K_stream");
        #pragma HLS STREAM variable=K_stream depth=FIFO_DEPTH
        
        hls::stream<v_row_t> V_stream("V_stream");
        #pragma HLS STREAM variable=V_stream depth=FIFO_DEPTH
        
        hls::stream<score_packet_t> score_stream("score_stream");
        #pragma HLS STREAM variable=score_stream depth=FIFO_DEPTH
        
        hls::stream<softmax_packet_t> softmax_stream("softmax_stream");
        #pragma HLS STREAM variable=softmax_stream depth=FIFO_DEPTH
        
        #pragma HLS DATAFLOW
        
        stage_load_Q((int8_t*)Q, Q_scale, Q_stream, i);
        stage_load_K((int8_t*)K, K_scale, K_stream, num_kv_blocks);
        stage_load_V((int8_t*)V, V_scale, V_stream, num_kv_blocks);
        stage_compute_scores(Q_stream, K_stream, score_stream, num_kv_blocks);
        stage_softmax_update(score_stream, softmax_stream, local_m, local_l, num_kv_blocks);
        stage_accumulate_output(softmax_stream, V_stream, local_O, num_kv_blocks);
        stage_normalize_store((fixed_t*)Output, local_O, local_l, i);
    }
}
