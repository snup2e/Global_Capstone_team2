#include "dcl_optimized.h"
#include <cmath>
#include <cstring>

using namespace std;

// --------------------------------------------------------
// INT8 텐서 로드 함수
// --------------------------------------------------------
void load_tensor_int8(const char* filename, int8_t* tensor, int size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    size_t elements_read = fread(tensor, sizeof(int8_t), size, file);
    if (elements_read != (size_t)size) {
        fprintf(stderr, "Error reading file: %s (read %zu, expected %d)\n", 
                filename, elements_read, size);
        fclose(file);
        exit(1);
    }

    fclose(file);
}

// --------------------------------------------------------
// Scale factor 로드 함수
// --------------------------------------------------------
void load_scale(const char* filename, float* scale, int size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    size_t elements_read = fread(scale, sizeof(float), size, file);
    if (elements_read != (size_t)size) {
        fprintf(stderr, "Error reading file: %s (read %zu, expected %d)\n", 
                filename, elements_read, size);
        fclose(file);
        exit(1);
    }

    fclose(file);
}

// --------------------------------------------------------
// Reference output 로드 (fixed_t 형식)
// --------------------------------------------------------
void load_tensor_fixed(const char* filename, fixed_t* tensor, int size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    size_t elements_read = fread(tensor, sizeof(fixed_t), size, file);
    if (elements_read != (size_t)size) {
        fprintf(stderr, "Error reading file: %s (read %zu, expected %d)\n", 
                filename, elements_read, size);
        fclose(file);
        exit(1);
    }

    fclose(file);
}

// --------------------------------------------------------
// FP32 Reference Attention (검증용)
// --------------------------------------------------------
void reference_attention_fp32(
    int8_t Q[N][dk], int8_t K[N][dk], int8_t V[N][dv],
    float Q_scale[N], float K_scale[N], float V_scale[N],
    float Output_ref[N][dv]
) {
    float scale = 1.0f / sqrtf((float)dk);
    
    // Attention scores: S = Q * K^T * scale
    float S[N][N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < dk; k++) {
                // Dequantize: Q_real = Q_int8 * Q_scale, K_real = K_int8 * K_scale
                float q_val = (float)Q[i][k] * Q_scale[i];
                float k_val = (float)K[j][k] * K_scale[j];
                sum += q_val * k_val;
            }
            S[i][j] = sum * scale;
        }
    }
    
    // Softmax (row-wise)
    float P[N][N];
    for (int i = 0; i < N; i++) {
        // Find max for numerical stability
        float max_val = S[i][0];
        for (int j = 1; j < N; j++) {
            if (S[i][j] > max_val) max_val = S[i][j];
        }
        
        // Exp and sum
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            P[i][j] = expf(S[i][j] - max_val);
            sum += P[i][j];
        }
        
        // Normalize
        for (int j = 0; j < N; j++) {
            P[i][j] /= sum;
        }
    }
    
    // Output: O = P * V
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dv; d++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                // Dequantize V: V_real = V_int8 * V_scale
                float v_val = (float)V[j][d] * V_scale[j];
                sum += P[i][j] * v_val;
            }
            Output_ref[i][d] = sum;
        }
    }
}

// --------------------------------------------------------
// Main
// --------------------------------------------------------
int main() {
    printf("==============================================\n");
    printf("Flash Attention INT8 Testbench\n");
    printf("N=%d, dk=%d, dv=%d\n", N, dk, dv);
    printf("==============================================\n\n");

    // --------------------------------------------------------
    // 메모리 할당
    // --------------------------------------------------------
    int8_t Q[N][dk];
    int8_t K[N][dk];
    int8_t V[N][dv];
    
    float Q_scale[N];
    float K_scale[N];
    float V_scale[N];
    
    fixed_t Output_HLS[N][dv];
    float Output_ref[N][dv];

    // --------------------------------------------------------
    // 파일에서 로드 또는 테스트 데이터 생성
    // --------------------------------------------------------
    bool use_file = false;  // 파일이 있으면 true로 변경
    
    if (use_file) {
        printf("Loading tensors from files...\n");
        load_tensor_int8("Q_int8.bin", (int8_t*)Q, N * dk);
        load_tensor_int8("K_int8.bin", (int8_t*)K, N * dk);
        load_tensor_int8("V_int8.bin", (int8_t*)V, N * dv);
        load_scale("Q_scale.bin", Q_scale, N);
        load_scale("K_scale.bin", K_scale, N);
        load_scale("V_scale.bin", V_scale, N);
    } else {
        printf("Generating random test data...\n");
        srand(42);  // 재현성을 위한 시드 고정
        
        // 랜덤 INT8 데이터 생성
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < dk; k++) {
                Q[i][k] = (int8_t)(rand() % 256 - 128);  // [-128, 127]
                K[i][k] = (int8_t)(rand() % 256 - 128);
            }
            for (int v = 0; v < dv; v++) {
                V[i][v] = (int8_t)(rand() % 256 - 128);
            }
            
            // 스케일 팩터 (일반적으로 작은 값)
            Q_scale[i] = 0.02f + (rand() % 100) * 0.0005f;
            K_scale[i] = 0.02f + (rand() % 100) * 0.0005f;
            V_scale[i] = 0.02f + (rand() % 100) * 0.0005f;
        }
    }

    // --------------------------------------------------------
    // HLS 출력 초기화
    // --------------------------------------------------------
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dv; d++) {
            Output_HLS[i][d] = 0;
        }
    }

    // --------------------------------------------------------
    // Reference 계산 (FP32)
    // --------------------------------------------------------
    printf("Computing reference attention (FP32)...\n");
    reference_attention_fp32(Q, K, V, Q_scale, K_scale, V_scale, Output_ref);

    // --------------------------------------------------------
    // HLS 커널 호출
    // --------------------------------------------------------
    printf("Running HLS kernel...\n");
    compute_attention_HLS(Q, K, V, Q_scale, K_scale, V_scale, Output_HLS);

    // --------------------------------------------------------
    // 결과 비교
    // --------------------------------------------------------
    printf("\nComparing results...\n");
    
    double mse = 0.0;
    double max_error = 0.0;
    int max_error_i = 0, max_error_d = 0;
    
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dv; d++) {
            float hls_val = Output_HLS[i][d].to_float();
            float ref_val = Output_ref[i][d];
            float error = hls_val - ref_val;
            
            mse += error * error;
            
            if (fabs(error) > max_error) {
                max_error = fabs(error);
                max_error_i = i;
                max_error_d = d;
            }
        }
    }
    
    mse /= (N * dv);
    double rmse = sqrt(mse);
    
    printf("\n==============================================\n");
    printf("Results:\n");
    printf("==============================================\n");
    printf("MSE:        %.8f\n", mse);
    printf("RMSE:       %.8f\n", rmse);
    printf("Max Error:  %.8f at [%d][%d]\n", max_error, max_error_i, max_error_d);
    printf("  HLS:  %.8f\n", Output_HLS[max_error_i][max_error_d].to_float());
    printf("  Ref:  %.8f\n", Output_ref[max_error_i][max_error_d]);
    
    // 샘플 출력
    printf("\nSample outputs (first 5 rows, first 5 cols):\n");
    printf("%-12s %-12s %-12s\n", "HLS", "Reference", "Diff");
    for (int i = 0; i < 5; i++) {
        for (int d = 0; d < 5; d++) {
            float hls_val = Output_HLS[i][d].to_float();
            float ref_val = Output_ref[i][d];
            printf("[%d][%d] %-10.6f %-10.6f %-10.6f\n", 
                   i, d, hls_val, ref_val, hls_val - ref_val);
        }
    }
    
    // Pass/Fail 판정
    printf("\n==============================================\n");
    if (rmse < 0.1) {
        printf("TEST PASSED (RMSE < 0.1)\n");
    } else if (rmse < 0.5) {
        printf("TEST MARGINAL (0.1 <= RMSE < 0.5)\n");
    } else {
        printf("TEST FAILED (RMSE >= 0.5)\n");
    }
    printf("==============================================\n");

    return (rmse < 0.5) ? 0 : 1;
}
