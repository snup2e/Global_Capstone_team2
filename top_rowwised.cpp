#include "dcl_new.h"


void softmax_1row(fixed_t data[N]) {
    #pragma HLS INLINE


    ap_fixed<32, 16> max_val = data[0];
    FIND_MAX: for (int i = 1; i < N; ++i) {
        #pragma HLS PIPELINE II=1
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    ap_fixed<32, 16> sum = 0;
    CALC_EXP: for (int i = 0; i < N; ++i) {
        #pragma HLS PIPELINE II=1
        data[i] = hls::exp(data[i] - max_val);
        sum += data[i];
    }

    ap_fixed<32, 16> inv_sum = ap_fixed<32, 16>(1.0) / sum; 
    NORMALIZE: for (int i = 0; i < N; ++i) {
        #pragma HLS PIPELINE II=1
        data[i] = data[i] * inv_sum;

    }
}

void compute_attention_HLS(fixed_t Q[N][dk], fixed_t K[N][dk], fixed_t V[N][dv], fixed_t Output[N][dv]) {

    #pragma HLS interface m_axi port=Q offset=slave bundle=mem1 depth=N*dk
    #pragma HLS interface m_axi port=K offset=slave bundle=mem1 depth=N*dk
    #pragma HLS interface m_axi port=V offset=slave bundle=mem1 depth=N*dv
    #pragma HLS interface m_axi port=Output offset=slave bundle=mem2 depth=N*dv
    #pragma HLS interface s_axilite port=return


    fixed_t local_K[N][dk];
    fixed_t local_V[N][dv];
    
 
    fixed_t curr_Q[dk];      
    fixed_t att_score[N];    
    fixed_t curr_out[dv];   



    #pragma HLS ARRAY_PARTITION variable=local_K dim=2 factor=8 cyclic
    #pragma HLS ARRAY_PARTITION variable=local_V dim=2 factor=8 cyclic
    
    #pragma HLS ARRAY_PARTITION variable=curr_Q dim=1 factor=8 cyclic
    #pragma HLS ARRAY_PARTITION variable=att_score dim=1 factor=16 cyclic
    #pragma HLS ARRAY_PARTITION variable=curr_out dim=1 factor=8 cyclic



    LOAD_K: for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE II=1
        for(int j=0; j<dk; j++) {
            local_K[i][j] = K[i][j];
        }
    }


    LOAD_V: for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE II=1
        for(int j=0; j<dv; j++) {
            local_V[i][j] = V[i][j];
        }
    }

    fixed_t scale = 1.0 / sqrt((float)dk);


    ROW_LOOP: for (int i = 0; i < N; ++i) {
        

        LOAD_Q_ROW: for(int j=0; j<dk; j++) {
            #pragma HLS PIPELINE II=1
            curr_Q[j] = Q[i][j];
        }


        CALC_SCORE: for (int j = 0; j < N; ++j) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 16> sum = 0;
            DOT_PROD_K: for (int k = 0; k < dk; ++k) {
                sum += curr_Q[k] * local_K[j][k];
            }
            att_score[j] = sum * scale;
        }


        softmax_1row(att_score);

        CALC_OUT: for (int j = 0; j < dv; ++j) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 16> sum = 0;
            DOT_PROD_V: for (int k = 0; k < N; ++k) {
                sum += att_score[k] * local_V[k][j];
            }
            curr_out[j] = sum;
        }

        STORE_OUT: for(int j=0; j<dv; j++) {
            #pragma HLS PIPELINE II=1
            Output[i][j] = curr_out[j];
        }
    }
}