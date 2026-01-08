#include "dcl_new.h"
#include <hls_math.h>

// Br: Block size of Query
// Bc: Block size of Key/Value
#define Br 64 
#define Bc 64 

void compute_attention_HLS(
    fixed_t Q[N][dk], 
    fixed_t K[N][dk], 
    fixed_t V[N][dv], 
    fixed_t Output[N][dv]
) {
    #pragma HLS INTERFACE mode=m_axi port=Q bundle=gmem0 depth=N*dk
    #pragma HLS INTERFACE mode=m_axi port=K bundle=gmem1 depth=N*dk
    #pragma HLS INTERFACE mode=m_axi port=V bundle=gmem2 depth=N*dv
    #pragma HLS INTERFACE mode=m_axi port=Output bundle=gmem3 depth=N*dv
    
    // Local buffers with partitioning for parallel access
    fixed_t local_Q[Br][dk];
    #pragma HLS ARRAY_PARTITION variable=local_Q cyclic factor=4 dim=2
    
    fixed_t local_K[Bc][dk];
    #pragma HLS ARRAY_PARTITION variable=local_K cyclic factor=4 dim=2
    
    fixed_t local_V[Bc][dv];
    #pragma HLS ARRAY_PARTITION variable=local_V cyclic factor=4 dim=2
    
    // Accumulation buffers (higher precision)
    ap_fixed<32,16> local_O[Br][dv];
    #pragma HLS ARRAY_PARTITION variable=local_O cyclic factor=4 dim=2
    
    ap_fixed<32,16> local_m[Br];
    #pragma HLS ARRAY_PARTITION variable=local_m complete
    
    ap_fixed<32,16> local_l[Br];
    #pragma HLS ARRAY_PARTITION variable=local_l complete

    fixed_t scale = 1.0 / hls::sqrt((float)dk);

    // Outer loop 1: iterate over Q blocks
    OUTER_Q_LOOP:
    for (int i = 0; i < N; i += Br) {
        
        // Load Q block
        LOAD_Q_ROW:
        for (int r = 0; r < Br; r++) {
            #pragma HLS PIPELINE II=1
            LOAD_Q_COL:
            for (int k = 0; k < dk; k++) {
                local_Q[r][k] = Q[i + r][k];
            }
        }

        // Initialize statistics
        INIT_STATS:
        for (int r = 0; r < Br; r++) {
            #pragma HLS UNROLL
            local_m[r] = -10000.0; 
            local_l[r] = 0;
            INIT_OUTPUT:
            for (int c = 0; c < dv; c++) {
                #pragma HLS PIPELINE II=1
                local_O[r][c] = 0; 
            }
        }

        // Outer loop 2: iterate over K,V blocks
        OUTER_KV_LOOP:
        for (int j = 0; j < N; j += Bc) {

            // Load K and V blocks
            LOAD_KV:
            for (int c = 0; c < Bc; c++) {
                #pragma HLS PIPELINE II=1
                LOAD_K:
                for (int k = 0; k < dk; k++) {
                    local_K[c][k] = K[j + c][k];
                }
                LOAD_V:
                for (int v = 0; v < dv; v++) {
                    local_V[c][v] = V[j + c][v];
                }
            }

            // Process each row in Q block
            PROCESS_ROW:
            for (int r = 0; r < Br; r++) {
                
                // Score calculation with higher precision
                ap_fixed<32,16> scores[Bc];
                #pragma HLS ARRAY_PARTITION variable=scores complete
                
                ap_fixed<32,16> row_max_val = -10000.0;

                // Compute Q*K^T scores
                SCORE_LOOP:
                for (int c = 0; c < Bc; c++) {
                    #pragma HLS PIPELINE II=1
                    ap_fixed<32,16> score_sum = 0;
                    SCORE_DOT:
                    for (int k = 0; k < dk; k++) {
                        #pragma HLS UNROLL factor=4
                        score_sum += (ap_fixed<32,16>)local_Q[r][k] * (ap_fixed<32,16>)local_K[c][k];
                    }
                    scores[c] = score_sum * scale;

                    if (scores[c] > row_max_val) {
                        row_max_val = scores[c];
                    }
                }

                // Online softmax update
                ap_fixed<32,16> m_prev = local_m[r];
                ap_fixed<32,16> m_new = (m_prev > row_max_val) ? m_prev : row_max_val;
                
                ap_fixed<32,16> correction_prev = hls::exp((float)(m_prev - m_new));

                // Compute softmax probabilities
                ap_fixed<32,16> p_sum_curr = 0;
                ap_fixed<32,16> P[Bc];
                #pragma HLS ARRAY_PARTITION variable=P complete

                SOFTMAX_LOOP:
                for (int c = 0; c < Bc; c++) {
                    #pragma HLS PIPELINE II=1
                    P[c] = hls::exp((float)(scores[c] - m_new)); 
                    p_sum_curr += P[c];
                }

                local_l[r] = local_l[r] * correction_prev + p_sum_curr;
                local_m[r] = m_new;

                // Update output (P*V)
                OUTPUT_UPDATE:
                for (int v = 0; v < dv; v++) {
                    #pragma HLS PIPELINE II=1
                    ap_fixed<32,16> weighted_sum = 0;
                    WEIGHTED_SUM:
                    for (int c = 0; c < Bc; c++) {
                        #pragma HLS UNROLL factor=4
                        weighted_sum += P[c] * (ap_fixed<32,16>)local_V[c][v];
                    }
                    local_O[r][v] = local_O[r][v] * correction_prev + weighted_sum;
                }
            }
        }

        // Final normalization and write back
        WRITE_OUTPUT:
        for (int r = 0; r < Br; r++) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32,16> inv_sum = ap_fixed<32, 16>(1.0) / local_l[r];
            NORMALIZE:
            for (int v = 0; v < dv; v++) {
                Output[i + r][v] = (fixed_t)(local_O[r][v] * inv_sum);
            }
        }
    }
}