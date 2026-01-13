#include "dcl_optimized.h"

// KV260 bus width: 128-bit / 8-bit = 16
#define BUS_PACK_FACTOR 16 

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


    qint8_t local_Q[Br][dk];
    #pragma HLS ARRAY_PARTITION variable=local_Q cyclic factor=4 dim=2
    
    qint8_t local_K[Bc][dk];
    #pragma HLS ARRAY_PARTITION variable=local_K cyclic factor=4 dim=2
    
    qint8_t local_V[Bc][dv];
    #pragma HLS ARRAY_PARTITION variable=local_V cyclic factor=4 dim=2
    
    scale_fixed_t local_scale_Q[Br];
    scale_fixed_t local_scale_K[Bc];
    scale_fixed_t local_scale_V[Bc];
    
    ap_fixed<32,16> local_O[Br][dv];
    #pragma HLS ARRAY_PARTITION variable=local_O cyclic factor=4 dim=2 

    ap_fixed<32,16> local_m[Br];
    #pragma HLS ARRAY_PARTITION variable=local_m complete
    ap_fixed<32,16> local_l[Br];
    #pragma HLS ARRAY_PARTITION variable=local_l complete


    OUTER_Q_LOOP:
    for (int i = 0; i < N; i += Br) {
        
        LOAD_Q_SCALE:
        for (int r = 0; r < Br; r++) {
            #pragma HLS PIPELINE II=1
            local_scale_Q[r] = (scale_fixed_t)scale_Q[i + r];
        }

        LOAD_Q_MATRIX:
        for (int r = 0; r < Br; r++) {
            for (int k = 0; k < dk; k++) {
                #pragma HLS PIPELINE II=1
                local_Q[r][k] = Q[i + r][k];
            }
        }

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

        OUTER_KV_LOOP:
        for (int j = 0; j < N; j += Bc) {
         
            // --- LOAD K PART ---
            LOAD_K_SCALE:
            for (int c = 0; c < Bc; c++) {
                 #pragma HLS PIPELINE II=1
                 local_scale_K[c] = (scale_fixed_t)scale_K[j + c];
            }

            LOAD_K_MATRIX:
            for (int c = 0; c < Bc; c++) {
                for (int k = 0; k < dk; k++) {
                    #pragma HLS PIPELINE II=1
                    local_K[c][k] = K[j + c][k];
                }
            }

            // --- LOAD V PART ---
            LOAD_V_SCALE:
            for (int c = 0; c < Bc; c++) {
                #pragma HLS PIPELINE II=1
                local_scale_V[c] = (scale_fixed_t)scale_V[j + c];
            }

            LOAD_V_MATRIX:
            for (int c = 0; c < Bc; c++) {
                for (int v = 0; v < dv; v++) {
                    #pragma HLS PIPELINE II=1
                    local_V[c][v] = V[j + c][v];
                }
            }
            // -------------------

            PROCESS_ROW:
            for (int r = 0; r < Br; r++) {

                ap_fixed<32,16> scores[Bc];
                #pragma HLS ARRAY_PARTITION variable=scores complete
                ap_fixed<32,16> row_max_val = -10000.0;

                SCORE_LOOP:
                for (int c = 0; c < Bc; c++) {
                    qint32_t score_sum_int = 0;
                    
                    SCORE_DOT:
                    for (int k = 0; k < dk; k++) {
                        #pragma HLS PIPELINE II=1 
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
                PRE_SCALE_LOOP:
                for (int c = 0; c < Bc; c++) {
                    #pragma HLS PIPELINE II=1 
                    scaled_P[c] = P[c] * local_scale_V[c];
                }
                
                local_l[r] = local_l[r] * correction_prev + p_sum_curr;
                local_m[r] = m_new;

                OUTPUT_UPDATE:
                for (int v = 0; v < dv; v++) {
                    ap_fixed<32,16> weighted_sum = 0;
                    
                    WEIGHTED_SUM:
                    for (int c = 0; c < Bc; c++) {
                        #pragma HLS PIPELINE II=1
                        auto term = scaled_P[c] * local_V[c][v];
                        weighted_sum += term;
                    }
                    local_O[r][v] = local_O[r][v] * correction_prev + weighted_sum;
                }
            }
        }


        WRITE_OUTPUT:
        for (int r = 0; r < Br; r++) {
            ap_fixed<32,16> inv_sum = ap_fixed<32, 16>(1.0) / local_l[r];
            
            for (int v = 0; v < dv; v++) {
                #pragma HLS PIPELINE II=1
                Output[i + r][v] = (fixed_t)(local_O[r][v] * inv_sum);
            }
        }
    }
}