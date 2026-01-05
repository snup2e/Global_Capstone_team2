#include "dcl_new.h"

using namespace std;

void load_tensor(const char* filename, fixed_t tensor[][dk], int D) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    size_t elements_read = fread(tensor, sizeof(fixed_t), N * D, file);
    if (elements_read != N * D) {
        fprintf(stderr, "Error reading file: %s\n", filename);
        fclose(file);
        exit(1);
    }

    fclose(file);
}


int main() {
    // Allocate memory for tensors
    fixed_t Q[N][dk];
    fixed_t K[N][dk];
    fixed_t V[N][dv];
    fixed_t Output_ref[N][dv];
    fixed_t Output_HLS[N][dv];

    // Load tensors from binary files
    load_tensor("Q_tensor.bin", Q, dk);
    load_tensor("K_tensor.bin", K, dk);
    load_tensor("V_tensor.bin", V, dv);
    load_tensor("Output_tensor.bin", Output_ref, dv);

    for(int j = 0; j < N; j++) {
        for(int k = 0; k < dv; k++) {
            Output_HLS[j][k] = 0;
        }
    }

    // call HLS kernel
    compute_attention_HLS(Q, K, V, Output_HLS);
    
    float error = 0;
    // compare HLS output and reference output tensor
    for(int j = 0; j < N; j++) {
        for(int k = 0; k < dv; k++) {
            error += std::pow(Output_HLS[j][k].to_float() - Output_ref[j][k].to_float(), 2);
            //printf("Output_HLS[%d][%d]: %.8f; Output_ref[%d][%d]: %.8f\n", 
            //j, k, Output_HLS[j][k].to_float(), j, k, Output_ref[j][k].to_float());
        }
    }
    error = error / (N * dv);
    printf("MSE: %.8f\n", error);

    return 0;
}