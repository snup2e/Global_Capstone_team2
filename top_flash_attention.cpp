#include "dcl_new.h"
#include <hls_math.h> // for hls::exp, hls::sqrt 


// Br: Block size of Query
// Bc: Block size of Key/Value
#define Br 64 
#define Bc 64 

void compute_flash_attention(
    fixed_t Q[N][dk], 
    fixed_t K[N][dk], 
    fixed_t V[N][dv], 
    fixed_t Output[N][dv]
) {
    //local buffer
    fixed_t local_Q[Br][dk];
    fixed_t local_K[Bc][dk];
    fixed_t local_V[Bc][dv];
    
    fixed_t local_O[Br][dv];   
    fixed_t local_m[Br];       
    fixed_t local_l[Br];       

    fixed_t scale = 1.0 / hls::sqrt((float)dk);

    //outer loop1
    for (int i = 0; i < N; i += Br) {
        
        for (int r = 0; r < Br; r++) {
            for (int k = 0; k < dk; k++) {
                local_Q[r][k] = Q[i + r][k];
            }
        }

        for (int r = 0; r < Br; r++) {
            local_m[r] = -10000.0; 
            local_l[r] = 0;        
            for (int c = 0; c < dv; c++) {
                local_O[r][c] = 0; 
            }
        }

        //outer loop2
        for (int j = 0; j < N; j += Bc) {

            for (int c = 0; c < Bc; c++) {
                for (int k = 0; k < dk; k++) {
                    local_K[c][k] = K[j + c][k];
                }
                for (int v = 0; v < dv; v++) {
                    local_V[c][v] = V[j + c][v];
                }
            }

 
            for (int r = 0; r < Br; r++) {
                
                fixed_t scores[Bc];
                fixed_t row_max_val = -10000.0;

                for (int c = 0; c < Bc; c++) {
                    fixed_t score_sum = 0;
                    for (int k = 0; k < dk; k++) {
                        score_sum += local_Q[r][k] * local_K[c][k];
                    }
                    scores[c] = score_sum * scale; 

                    if (scores[c] > row_max_val) {
                        row_max_val = scores[c];
                    }
                }


                fixed_t m_prev = local_m[r];
                fixed_t m_new = (m_prev > row_max_val) ? m_prev : row_max_val;
                
                fixed_t correction_prev = hls::exp(m_prev - m_new);


                fixed_t p_sum_curr = 0;
                fixed_t P[Bc]; 

                for (int c = 0; c < Bc; c++) {

                    P[c] = hls::exp(scores[c] - m_new); 
                    p_sum_curr += P[c];
                }

                local_l[r] = local_l[r] * correction_prev + p_sum_curr;
                local_m[r] = m_new; 

                for (int v = 0; v < dv; v++) {
                    fixed_t weighted_sum = 0;
                    for (int c = 0; c < Bc; c++) {
                        weighted_sum += P[c] * local_V[c][v];
                    }
                    local_O[r][v] = local_O[r][v] * correction_prev + weighted_sum;
                }
            }
        }

        for (int r = 0; r < Br; r++) {
            fixed_t inv_sum = 1.0 / local_l[r];
            for (int v = 0; v < dv; v++) {
                Output[i + r][v] = local_O[r][v] * inv_sum;
            }
        }
    }
}