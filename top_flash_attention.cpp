#include "dcl_new.h"
#include "hls_stream.h"

// Block sizes
#define Br 32 
#define Bc 32 
#define FIFO_DEPTH 4

// --------------------------------------------------------
// Stream 타입 정의
// --------------------------------------------------------
typedef struct {
    fixed_t data[Bc][dk];
} K_block_t;

typedef struct {
    fixed_t data[Bc][dv];
} V_block_t;

typedef struct {
    ap_fixed<32,16> scores[Bc];
    ap_fixed<32,16> row_max;
    int row_idx;
} score_packet_t;

typedef struct {
    ap_fixed<32,16> P[Bc];
    ap_fixed<32,16> m_new;
    ap_fixed<32,16> correction;
    ap_fixed<32,16> p_sum;
    int row_idx;
} softmax_packet_t;

// --------------------------------------------------------
// Stage 1: Q 블록 로드 및 초기화
// --------------------------------------------------------
void stage_load_Q(
    fixed_t* Q,
    hls::stream<fixed_t[dk]> &Q_stream,
    ap_fixed<32,16> local_m[Br],
    ap_fixed<32,16> local_l[Br],
    int i_offset,
    int &stage_done
) {
    #pragma HLS INLINE off
    
    // Q 로드
    for (int r = 0; r < Br; r++) {
        #pragma HLS PIPELINE II=1
        fixed_t q_row[dk];
        #pragma HLS ARRAY_PARTITION variable=q_row complete
        
        for (int k = 0; k < dk; k++) {
            q_row[k] = Q[(i_offset + r) * dk + k];
        }
        Q_stream.write(q_row);
        
        // 초기화
        local_m[r] = -10000.0;
        local_l[r] = 0.0;
    }
    stage_done = 1;
}

// --------------------------------------------------------
// Stage 2: K 블록 로드 및 브로드캐스트
// --------------------------------------------------------
void stage_load_K(
    fixed_t* K,
    hls::stream<K_block_t> &K_stream,
    int num_blocks
) {
    #pragma HLS INLINE off
    
    LOAD_K_BLOCKS:
    for (int j = 0; j < num_blocks; j++) {
        K_block_t k_block;
        #pragma HLS ARRAY_PARTITION variable=k_block.data cyclic factor=4 dim=2
        
        for (int c = 0; c < Bc; c++) {
            for (int k = 0; k < dk; k++) {
                #pragma HLS PIPELINE II=1
                k_block.data[c][k] = K[(j * Bc + c) * dk + k];
            }
        }
        K_stream.write(k_block);
    }
}

// --------------------------------------------------------
// Stage 3: V 블록 로드 및 브로드캐스트  
// --------------------------------------------------------
void stage_load_V(
    fixed_t* V,
    hls::stream<V_block_t> &V_stream,
    int num_blocks
) {
    #pragma HLS INLINE off
    
    LOAD_V_BLOCKS:
    for (int j = 0; j < num_blocks; j++) {
        V_block_t v_block;
        #pragma HLS ARRAY_PARTITION variable=v_block.data cyclic factor=4 dim=2
        
        for (int c = 0; c < Bc; c++) {
            for (int v = 0; v < dv; v++) {
                #pragma HLS PIPELINE II=1
                v_block.data[c][v] = V[(j * Bc + c) * dv + v];
            }
        }
        V_stream.write(v_block);
    }
}

// --------------------------------------------------------
// Stage 4: Score 계산 (Q * K^T) - 최적화 버전
// --------------------------------------------------------
void stage_compute_scores(
    hls::stream<fixed_t[dk]> &Q_stream,
    hls::stream<K_block_t> &K_stream,
    hls::stream<score_packet_t> &score_stream,
    int num_kv_blocks
) {
    #pragma HLS INLINE off
    
    fixed_t Q_buffer[Br][dk];
    #pragma HLS ARRAY_PARTITION variable=Q_buffer cyclic factor=4 dim=2
    
    // Q 읽기
    LOAD_Q:
    for (int r = 0; r < Br; r++) {
        #pragma HLS PIPELINE II=1
        fixed_t q_row[dk];
        Q_stream.read(q_row);
        for (int k = 0; k < dk; k++) {
            Q_buffer[r][k] = q_row[k];
        }
    }
    
    fixed_t scale = 1.0 / hls::sqrt((float)dk);
    
    // 각 K 블록에 대해
    KV_BLOCKS:
    for (int j = 0; j < num_kv_blocks; j++) {
        // Stream에서 K 블록 읽기
        K_block_t k_block_in = K_stream.read();
        
        // 로컬 파티션된 버퍼로 복사 (메모리 액세스 최적화)
        fixed_t local_K[Bc][dk];
        #pragma HLS ARRAY_PARTITION variable=local_K cyclic factor=4 dim=2
        
        COPY_K:
        for (int c = 0; c < Bc; c++) {
            for (int k = 0; k < dk; k++) {
                #pragma HLS PIPELINE II=1
                local_K[c][k] = k_block_in.data[c][k];
            }
        }
        
        // 각 Q 행에 대해 Score 계산
        Q_ROWS:
        for (int r = 0; r < Br; r++) {
            score_packet_t pkt;
            pkt.row_max = -10000.0;
            pkt.row_idx = r;
            
            // Dot product: Q[r] * K[c]^T for all c
            COMPUTE_SCORES:
            for (int c = 0; c < Bc; c++) {
                #pragma HLS PIPELINE II=1
                // 주의: 여기에 PIPELINE을 걸어서 c 루프를 파이프라인화
                // 바깥 Q_ROWS 루프의 PIPELINE은 제거
                
                ap_fixed<32,16> score_sum = 0;
                #pragma HLS BIND_OP variable=score_sum op=add impl=dsp
                
                DOT_PRODUCT:
                for (int k = 0; k < dk; k++) {
                    #pragma HLS UNROLL factor=4
                    score_sum += (ap_fixed<32,16>)Q_buffer[r][k] * 
                                 (ap_fixed<32,16>)local_K[c][k];
                }
                
                pkt.scores[c] = score_sum * scale;
                
                // Row-wise max 추적
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
    
    SOFTMAX_BLOCKS:
    for (int j = 0; j < num_kv_blocks; j++) {
        SOFTMAX_ROWS:
        for (int r = 0; r < Br; r++) {
            #pragma HLS PIPELINE II=1
            
            score_packet_t score_pkt = score_stream.read();
            softmax_packet_t sm_pkt;
            sm_pkt.row_idx = score_pkt.row_idx;
            
            // Online softmax
            ap_fixed<32,16> m_prev = local_m[r];
            sm_pkt.m_new = (m_prev > score_pkt.row_max) ? m_prev : score_pkt.row_max;
            sm_pkt.correction = hls::exp((float)(m_prev - sm_pkt.m_new));
            
            sm_pkt.p_sum = 0;
            EXP_LOOP:
            for (int c = 0; c < Bc; c++) {
                #pragma HLS UNROLL factor=2
                sm_pkt.P[c] = hls::exp((float)(score_pkt.scores[c] - sm_pkt.m_new));
                sm_pkt.p_sum += sm_pkt.P[c];
            }
            
            local_l[r] = local_l[r] * sm_pkt.correction + sm_pkt.p_sum;
            local_m[r] = sm_pkt.m_new;
            
            softmax_stream.write(sm_pkt);
        }
    }
}

// --------------------------------------------------------
// Stage 6: Output 누적 (P * V) - 최적화 버전
// --------------------------------------------------------
void stage_accumulate_output(
    hls::stream<softmax_packet_t> &softmax_stream,
    hls::stream<V_block_t> &V_stream,
    ap_fixed<32,16> local_O[Br][dv],
    int num_kv_blocks
) {
    #pragma HLS INLINE off
    
    // O 초기화
    INIT_O:
    for (int r = 0; r < Br; r++) {
        for (int v = 0; v < dv; v++) {
            #pragma HLS PIPELINE II=1
            local_O[r][v] = 0;
        }
    }
    
    ACC_BLOCKS:
    for (int j = 0; j < num_kv_blocks; j++) {
        // V 블록 읽기 및 로컬 복사
        V_block_t v_block_in = V_stream.read();
        
        fixed_t local_V[Bc][dv];
        #pragma HLS ARRAY_PARTITION variable=local_V cyclic factor=4 dim=2
        
        COPY_V:
        for (int c = 0; c < Bc; c++) {
            for (int v = 0; v < dv; v++) {
                #pragma HLS PIPELINE II=1
                local_V[c][v] = v_block_in.data[c][v];
            }
        }
        
        ACC_ROWS:
        for (int r = 0; r < Br; r++) {
            softmax_packet_t sm_pkt = softmax_stream.read();
            
            // P * V 계산
            ACC_DIMS:
            for (int v = 0; v < dv; v++) {
                #pragma HLS PIPELINE II=1
                
                ap_fixed<32,16> weighted_sum = 0;
                #pragma HLS BIND_OP variable=weighted_sum op=add impl=dsp
                
                MAC_LOOP:
                for (int c = 0; c < Bc; c++) {
                    #pragma HLS UNROLL factor=4
                    weighted_sum += sm_pkt.P[c] * (ap_fixed<32,16>)local_V[c][v];
                }
                
                local_O[r][v] = local_O[r][v] * sm_pkt.correction + weighted_sum;
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
    
    // Statistics 버퍼 (전체 Q에 대해 유지)
    ap_fixed<32,16> local_m[Br];
    #pragma HLS ARRAY_PARTITION variable=local_m complete
    
    ap_fixed<32,16> local_l[Br];
    #pragma HLS ARRAY_PARTITION variable=local_l complete
    
    ap_fixed<32,16> local_O[Br][dv];
    #pragma HLS ARRAY_PARTITION variable=local_O cyclic factor=4 dim=2
    
    // Q 블록별 처리
    OUTER_Q_LOOP:
    for (int i = 0; i < N; i += Br) {
        #pragma HLS DATAFLOW
        
        // FIFO 스트림 선언
        hls::stream<fixed_t[dk]> Q_stream("Q_stream");
        #pragma HLS STREAM variable=Q_stream depth=FIFO_DEPTH
        
        hls::stream<K_block_t> K_stream("K_stream");
        #pragma HLS STREAM variable=K_stream depth=FIFO_DEPTH
        
        hls::stream<V_block_t> V_stream("V_stream");
        #pragma HLS STREAM variable=V_stream depth=FIFO_DEPTH
        
        hls::stream<score_packet_t> score_stream("score_stream");
        #pragma HLS STREAM variable=score_stream depth=FIFO_DEPTH
        
        hls::stream<softmax_packet_t> softmax_stream("softmax_stream");
        #pragma HLS STREAM variable=softmax_stream depth=FIFO_DEPTH
        
        int stage_done;
        
        // 7-stage pipeline
        stage_load_Q((fixed_t*)Q, Q_stream, local_m, local_l, i, stage_done);
        
        stage_load_K((fixed_t*)K, K_stream, num_kv_blocks);
        
        stage_load_V((fixed_t*)V, V_stream, num_kv_blocks);
        
        stage_compute_scores(Q_stream, K_stream, score_stream, num_kv_blocks);
        
        stage_softmax_update(score_stream, softmax_stream, local_m, local_l, num_kv_blocks);
        
        stage_accumulate_output(softmax_stream, V_stream, local_O, num_kv_blocks);
        
        stage_normalize_store((fixed_t*)Output, local_O, local_l, i);
    }
}