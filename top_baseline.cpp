#include "dcl_new.h"


void softmax_HLS(fixed_t matrix[N][N]) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=matrix dim=2 factor=16 cyclic

    for (int i = 0; i < N; ++i) {
        ap_fixed<32, 8> max_val = matrix[i][0];
        
        // Find MAX
        for (int j = 1; j < N; ++j) {
            #pragma HLS PIPELINE II=1
            if (matrix[i][j] > max_val) {
                max_val = matrix[i][j];
            }
        }
        
        // EXP & SUM
        ap_fixed<32, 8> sum = 0;
        for (int j = 0; j < N; ++j) {
            #pragma HLS PIPELINE II=1
            matrix[i][j] = hls::exp(matrix[i][j] - max_val);
            sum += matrix[i][j];
        }
        
        // Normalize
        for (int j = 0; j < N; ++j) {
            matrix[i][j] /= sum;
        }
    }
}

void compute_attention_HLS(fixed_t Q[N][dk], fixed_t K[N][dk], fixed_t V[N][dv], fixed_t Output[N][dv]) {

    #pragma HLS interface m_axi port=Q offset=slave bundle=mem1 depth=N*dk
    #pragma HLS interface m_axi port=K offset=slave bundle=mem1 depth=N*dk
    #pragma HLS interface m_axi port=V offset=slave bundle=mem1 depth=N*dv
    #pragma HLS interface m_axi port=Output offset=slave bundle=mem2 depth=N*dv
    #pragma HLS interface s_axilite port=return


    fixed_t local_Q[N][dk];
    fixed_t local_K[N][dk];
    fixed_t local_V[N][dv];
    fixed_t local_att[N][N];
    fixed_t local_out[N][dv];


    #pragma HLS ARRAY_PARTITION variable=local_Q dim=2 factor=8 cyclic
    #pragma HLS ARRAY_PARTITION variable=local_K dim=2 factor=8 cyclic
    #pragma HLS ARRAY_PARTITION variable=local_V dim=1 factor=8 cyclic 
    #pragma HLS ARRAY_PARTITION variable=local_att dim=2 factor=8 cyclic


    // LOAD_LOOP: for(int i=0; i<N; i++) {
    //     #pragma HLS PIPELINE II=1
    //     for(int j=0; j<dk; j++) local_Q[i][j] = Q[i][j];
    //     for(int j=0; j<dk; j++) local_K[i][j] = K[i][j];
    //     for(int j=0; j<dv; j++) local_V[i][j] = V[i][j];
    // }

    // 1. Load Q
    LOAD_Q_LOOP: for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE II=1
        for(int j=0; j<dk; j++) {
            local_Q[i][j] = Q[i][j];
        }
    }

    // 2. Load K
    LOAD_K_LOOP: for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE II=1
        for(int j=0; j<dk; j++) {
            local_K[i][j] = K[i][j];
        }
    }

    // 3. Load V
    LOAD_V_LOOP: for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE II=1
        for(int j=0; j<dv; j++) {
            local_V[i][j] = V[i][j];
        }
    }

    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);


    QK_ROWS: for (int i = 0; i < N; ++i) {
        QK_COLS: for (int j = 0; j < N; ++j) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < dk; ++k) {
                sum += local_Q[i][k] * local_K[j][k];
            }
            local_att[i][j] = sum * scale;
        }
    }

    // Softmax
    softmax_HLS(local_att);

    // Att * V
    AV_ROWS: for (int i = 0; i < N; ++i) {
        AV_COLS: for (int j = 0; j < dv; ++j) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += local_att[i][k] * local_V[k][j];
            }
            local_out[i][j] = sum;
        }
    }

    STORE_LOOP: for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE II=1
        for(int j=0; j<dv; j++) {
            Output[i][j] = local_out[i][j];
        }
    }
}
