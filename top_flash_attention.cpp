#include "dcl_new.h"
#include "hls_stream.h"

// Block sizes
#define Br 16 
#define Bc 16
#define FIFO_DEPTH 8

// --------------------------------------------------------
// Stream 타입 정의 - 행 단위로 변경하여 비트폭 제한 준수
// --------------------------------------------------------
typedef struct {
    fixed_t data[dk];
} row_dk_t;  // dk * 16 = 64 * 16 = 1024 bits (OK)

typedef struct {
    fixed_t data[dv];
} row_dv_t;  // dv * 16 = 64 * 16 = 1024 bits (OK)

typedef struct {
    ap_fixed<32,16> scores[Bc];
    ap_fixed<32,16> row_max;
    int row_idx;
    int block_idx;
} score_packet_t;  // Bc * 32 + 32 + 64 = 1120 bits (OK)

typedef struct {
    ap_fixed<32,16> P[Bc];
    ap_fixed<32,16> m_new;
    ap_fixed<32,16> correction;
    ap_fixed<32,16> p_sum;
    int row_idx;
    int block_idx;
} softmax_packet_t;  // Bc * 32 + 3*32 + 64 = 1184 bits (OK)

// --------------------------------------------------------
// Stage 1: Q 블록 로드
// --------------------------------------------------------
void stage_load_Q(
    fixed_t* Q,
    hls::stream<row_dk_t> &Q_stream,
    int i_offset
) {
    #pragma HLS INLINE off
    
    LOAD_Q_ROWS:
    for (int r = 0; r < Br; r++) {
        row_dk_t q_row;
        #pragma HLS ARRAY_PARTITION variable=q_row.data cyclic factor=8
        
        LOAD_Q_COLS:
        for (int k = 0; k < dk; k++) {
            #pragma HLS PIPELINE II=1
            q_row.data[k] = Q[(i_offset + r) * dk + k];
        }
        Q_stream.write(q_row);
    }
}

// --------------------------------------------------------
// Stage 2: K 블록 로드 (행 단위로 스트리밍)
// --------------------------------------------------------
void stage_load_K(
    fixed_t* K,
    hls::stream<row_dk_t> &K_stream,
    int num_blocks
) {
    #pragma HLS INLINE off
    
    LOAD_K_BLOCKS:
    for (int j = 0; j < num_blocks; j++) {
        LOAD_K_ROWS:
        for (int c = 0; c < Bc; c++) {
            row_dk_t k_row;
            #pragma HLS ARRAY_PARTITION variable=k_row.data cyclic factor=8
            
            LOAD_K_COLS:
            for (int k = 0; k < dk; k++) {
                #pragma HLS PIPELINE II=1
                k_row.data[k] = K[(j * Bc + c) * dk + k];
            }
            K_stream.write(k_row);
        }
    }
}

// --------------------------------------------------------
// Stage 3: V 블록 로드 (행 단위로 스트리밍)
// --------------------------------------------------------
void stage_load_V(
    fixed_t* V,
    hls::stream<row_dv_t> &V_stream,
    int num_blocks
) {
    #pragma HLS INLINE off
    
    LOAD_V_BLOCKS:
    for (int j = 0; j < num_blocks; j++) {
        LOAD_V_ROWS:
        for (int c = 0; c < Bc; c++) {
            row_dv_t v_row;
            #pragma HLS ARRAY_PARTITION variable=v_row.data cyclic factor=8
            
            LOAD_V_COLS:
            for (int v = 0; v < dv; v++) {
                #pragma HLS PIPELINE II=1
                v_row.data[v] = V[(j * Bc + c) * dv + v];
            }
            V_stream.write(v_row);
        }
    }
}

// --------------------------------------------------------
// Stage 4: Score 계산 (Q * K^T) 
// K를 행 단위로 받아서 로컬 버퍼에 축적 후 계산
// --------------------------------------------------------
void stage_compute_scores(
    hls::stream<row_dk_t> &Q_stream,
    hls::stream<row_dk_t> &K_stream,
    hls::stream<score_packet_t> &score_stream,
    int num_kv_blocks
) {
    #pragma HLS INLINE off
    
    // Q 버퍼
    fixed_t Q_buffer[Br][dk];
    #pragma HLS ARRAY_PARTITION variable=Q_buffer cyclic factor=8 dim=2
    
    // K 버퍼 (한 블록씩)
    fixed_t K_buffer[Bc][dk];
    #pragma HLS ARRAY_PARTITION variable=K_buffer cyclic factor=8 dim=2
    
    // Q 읽기
    LOAD_Q:
    for (int r = 0; r < Br; r++) {
        row_dk_t q_row = Q_stream.read();
        COPY_Q:
        for (int k = 0; k < dk; k++) {
            #pragma HLS UNROLL factor=8
            Q_buffer[r][k] = q_row.data[k];
        }
    }
    
    fixed_t scale = 1.0 / hls::sqrt((float)dk);
    
    // 각 K 블록에 대해
    KV_BLOCKS:
    for (int j = 0; j < num_kv_blocks; j++) {
        // K 블록 로드 (행 단위로 수신)
        RECV_K_ROWS:
        for (int c = 0; c < Bc; c++) {
            row_dk_t k_row = K_stream.read();
            COPY_K:
            for (int k = 0; k < dk; k++) {
                #pragma HLS UNROLL factor=8
                K_buffer[c][k] = k_row.data[k];
            }
        }
        
        // 각 Q 행에 대해 Score 계산
        Q_ROWS:
        for (int r = 0; r < Br; r++) {
            score_packet_t pkt;
            pkt.row_max = -10000.0;
            pkt.row_idx = r;
            pkt.block_idx = j;
            
            // Dot product: Q[r] * K[c]^T for all c
            COMPUTE_SCORES:
            for (int c = 0; c < Bc; c++) {
                #pragma HLS PIPELINE II=1
                
                ap_fixed<32,16> score_sum = 0;
                
                DOT_PRODUCT:
                for (int k = 0; k < dk; k++) {
                    #pragma HLS UNROLL factor=8
                    score_sum += (ap_fixed<32,16>)Q_buffer[r][k] * 
                                 (ap_fixed<32,16>)K_buffer[c][k];
                }
                
                pkt.scores[c] = score_sum * scale;
                
                if (pkt.scores[c] > pkt.row_max) {
                    pkt.row_max = pkt.scores[c];
                }
            }
            
            score_stream.write(pkt);
        }
    }
}

// --------------------------------------------------------
// Stage 5: Softmax 업데이트
// --------------------------------------------------------
void stage_softmax_update(
    hls::stream<score_packet_t> &score_stream,
    hls::stream<softmax_packet_t> &softmax_stream,
    ap_fixed<32,16> local_m[Br],
    ap_fixed<32,16> local_l[Br],
    int num_kv_blocks
) {
    #pragma HLS INLINE off
    
    // 초기화
    INIT_STATS:
    for (int r = 0; r < Br; r++) {
        #pragma HLS UNROLL
        local_m[r] = -10000.0;
        local_l[r] = 0.0;
    }
    
    SOFTMAX_BLOCKS:
    for (int j = 0; j < num_kv_blocks; j++) {
        SOFTMAX_ROWS:
        for (int r = 0; r < Br; r++) {
            score_packet_t score_pkt = score_stream.read();
            softmax_packet_t sm_pkt;
            sm_pkt.row_idx = score_pkt.row_idx;
            sm_pkt.block_idx = score_pkt.block_idx;
            
            int idx = score_pkt.row_idx;
            
            // Online softmax
            ap_fixed<32,16> m_prev = local_m[idx];
            sm_pkt.m_new = (m_prev > score_pkt.row_max) ? m_prev : score_pkt.row_max;
            sm_pkt.correction = hls::exp((float)(m_prev - sm_pkt.m_new));
            
            sm_pkt.p_sum = 0;
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
// V를 행 단위로 받아서 처리
// --------------------------------------------------------
void stage_accumulate_output(
    hls::stream<softmax_packet_t> &softmax_stream,
    hls::stream<row_dv_t> &V_stream,
    ap_fixed<32,16> local_O[Br][dv],
    int num_kv_blocks
) {
    #pragma HLS INLINE off
    
    // V 버퍼
    fixed_t V_buffer[Bc][dv];
    #pragma HLS ARRAY_PARTITION variable=V_buffer cyclic factor=8 dim=2
    
    // O 초기화
    INIT_O:
    for (int r = 0; r < Br; r++) {
        for (int v = 0; v < dv; v++) {
            #pragma HLS UNROLL factor=8
            local_O[r][v] = 0;
        }
    }
    
    ACC_BLOCKS:
    for (int j = 0; j < num_kv_blocks; j++) {
        // V 블록 로드 (행 단위로 수신)
        RECV_V_ROWS:
        for (int c = 0; c < Bc; c++) {
            row_dv_t v_row = V_stream.read();
            COPY_V:
            for (int v = 0; v < dv; v++) {
                #pragma HLS UNROLL factor=8
                V_buffer[c][v] = v_row.data[v];
            }
        }
        
        ACC_ROWS:
        for (int r = 0; r < Br; r++) {
            softmax_packet_t sm_pkt = softmax_stream.read();
            int idx = sm_pkt.row_idx;
            
            // P * V 계산
            ACC_DIMS:
            for (int v = 0; v < dv; v++) {
                #pragma HLS PIPELINE II=1
                
                ap_fixed<32,16> weighted_sum = 0;
                
                MAC_LOOP:
                for (int c = 0; c < Bc; c++) {
                    #pragma HLS UNROLL factor=8
                    weighted_sum += sm_pkt.P[c] * (ap_fixed<32,16>)V_buffer[c][v];
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
    ap_fixed<32,16> local_O[Br][dv],
    ap_fixed<32,16> local_l[Br],
    int i_offset
) {
    #pragma HLS INLINE off
    
    STORE_ROWS:
    for (int r = 0; r < Br; r++) {
        ap_fixed<32,16> inv_sum = ap_fixed<32,16>(1.0) / local_l[r];
        
        STORE_COLS:
        for (int v = 0; v < dv; v++) {
            #pragma HLS PIPELINE II=1
            Output[(i_offset + r) * dv + v] = (fixed_t)(local_O[r][v] * inv_sum);
        }
    }
}

// --------------------------------------------------------
// TOP FUNCTION - Pipelined Stage Architecture
// --------------------------------------------------------
void compute_attention_HLS(
    fixed_t Q[N][dk],
    fixed_t K[N][dk],
    fixed_t V[N][dv],
    fixed_t Output[N][dv]
) {
    #pragma HLS INTERFACE mode=m_axi port=Q bundle=gmem0 depth=N*dk max_read_burst_length=256
    #pragma HLS INTERFACE mode=m_axi port=K bundle=gmem1 depth=N*dk max_read_burst_length=256
    #pragma HLS INTERFACE mode=m_axi port=V bundle=gmem2 depth=N*dv max_read_burst_length=256
    #pragma HLS INTERFACE mode=m_axi port=Output bundle=gmem3 depth=N*dv max_write_burst_length=256
    
    const int num_kv_blocks = N / Bc;
    
    // Q 블록별 처리
    OUTER_Q_LOOP:
    for (int i = 0; i < N; i += Br) {
        // Statistics 버퍼 (각 Q 블록에 대해 새로 초기화)
        ap_fixed<32,16> local_m[Br];
        #pragma HLS ARRAY_PARTITION variable=local_m complete
        
        ap_fixed<32,16> local_l[Br];
        #pragma HLS ARRAY_PARTITION variable=local_l complete
        
        ap_fixed<32,16> local_O[Br][dv];
        #pragma HLS ARRAY_PARTITION variable=local_O cyclic factor=8 dim=2
        
        // FIFO 스트림 선언
        hls::stream<row_dk_t> Q_stream("Q_stream");
        #pragma HLS STREAM variable=Q_stream depth=FIFO_DEPTH
        
        hls::stream<row_dk_t> K_stream("K_stream");
        #pragma HLS STREAM variable=K_stream depth=FIFO_DEPTH
        
        hls::stream<row_dv_t> V_stream("V_stream");
        #pragma HLS STREAM variable=V_stream depth=FIFO_DEPTH
        
        hls::stream<score_packet_t> score_stream("score_stream");
        #pragma HLS STREAM variable=score_stream depth=FIFO_DEPTH
        
        hls::stream<softmax_packet_t> softmax_stream("softmax_stream");
        #pragma HLS STREAM variable=softmax_stream depth=FIFO_DEPTH
        
        #pragma HLS DATAFLOW
        
        // 7-stage pipeline
        stage_load_Q((fixed_t*)Q, Q_stream, i);
        
        stage_load_K((fixed_t*)K, K_stream, num_kv_blocks);
        
        stage_load_V((fixed_t*)V, V_stream, num_kv_blocks);
        
        stage_compute_scores(Q_stream, K_stream, score_stream, num_kv_blocks);
        
        stage_softmax_update(score_stream, softmax_stream, local_m, local_l, num_kv_blocks);
        
        stage_accumulate_output(softmax_stream, V_stream, local_O, num_kv_blocks);
        
        stage_normalize_store((fixed_t*)Output, local_O, local_l, i);
    }
}