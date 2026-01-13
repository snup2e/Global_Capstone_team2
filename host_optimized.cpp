#include "dcl_optimized.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

using namespace std;

// --------------------------------------------------------
// [표준 C 타입] INT8 텐서 로드 함수 (검증용/파일입력용)
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
// FP32 Reference Attention (검증용 - 표준 C 타입 사용)
// --------------------------------------------------------
void reference_attention_fp32(
    int8_t Q[N][dk], int8_t K[N][dk], int8_t V[N][dv],
    float Q_scale[N], float K_scale[N], float V_scale[N],
    float Output_ref[N][dv]
) {
    float scale = 1.0f / sqrtf((float)dk);
    
    for (int i = 0; i < N; i++) {
        // 1. Score 계산
        float scores[N];
        float max_val = -1e9;

        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < dk; k++) {
                float q_val = (float)Q[i][k] * Q_scale[i];
                float k_val = (float)K[j][k] * K_scale[j];
                sum += q_val * k_val;
            }
            scores[j] = sum * scale;
            if (scores[j] > max_val) max_val = scores[j];
        }
        
        // 2. Softmax
        float sum_exp = 0.0f;
        float P[N];
        for (int j = 0; j < N; j++) {
            P[j] = expf(scores[j] - max_val);
            sum_exp += P[j];
        }
        
        for (int j = 0; j < N; j++) {
            P[j] /= sum_exp;
        }
        
        // 3. Output Update
        for (int d = 0; d < dv; d++) {
            float sum_v = 0.0f;
            for (int j = 0; j < N; j++) {
                float v_val = (float)V[j][d] * V_scale[j];
                sum_v += P[j] * v_val;
            }
            Output_ref[i][d] = sum_v;
        }
    }
}

// --------------------------------------------------------
// Main
// --------------------------------------------------------
int main() {
    printf("==============================================\n");
    printf("Flash Attention INT8 Testbench (Fixed Type)\n");
    printf("N=%d, dk=%d, dv=%d\n", N, dk, dv);
    printf("==============================================\n\n");

    // --------------------------------------------------------
    // 1. [검증용] 표준 C 타입 변수 선언 (int8_t)
    // --------------------------------------------------------
    int8_t Q_ref[N][dk];
    int8_t K_ref[N][dk];
    int8_t V_ref[N][dv];
    float Output_ref[N][dv];

    // --------------------------------------------------------
    // 2. [HLS용] HLS 전용 타입 변수 선언 (qint8_t / ap_int<8>)
    // --------------------------------------------------------
    qint8_t Q_hls[N][dk];
    qint8_t K_hls[N][dk];
    qint8_t V_hls[N][dv];
    fixed_t Output_HLS[N][dv]; // 출력도 HLS 타입

    // Scales (공통)
    float Q_scale[N];
    float K_scale[N];
    float V_scale[N];
    
    // --------------------------------------------------------
    // 데이터 로드 또는 생성 (표준 C 타입 변수 사용)
    // --------------------------------------------------------
    bool use_file = false;  // 파일이 있으면 true로 변경
    
    if (use_file) {
        printf("Loading tensors from files...\n");
        load_tensor_int8("Q_int8.bin", (int8_t*)Q_ref, N * dk);
        load_tensor_int8("K_int8.bin", (int8_t*)K_ref, N * dk);
        load_tensor_int8("V_int8.bin", (int8_t*)V_ref, N * dv);
        load_scale("Q_scales.bin", Q_scale, N); 
        load_scale("K_scales.bin", K_scale, N);
        load_scale("V_scales.bin", V_scale, N);
    } else {
        printf("Generating random test data...\n");
        srand(42);
        
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < dk; k++) {
                Q_ref[i][k] = (int8_t)(rand() % 256 - 128);
                K_ref[i][k] = (int8_t)(rand() % 256 - 128);
            }
            for (int v = 0; v < dv; v++) {
                V_ref[i][v] = (int8_t)(rand() % 256 - 128);
            }
            
            Q_scale[i] = 0.02f + (rand() % 100) * 0.0005f;
            K_scale[i] = 0.02f + (rand() % 100) * 0.0005f;
            V_scale[i] = 0.02f + (rand() % 100) * 0.0005f;
        }
    }

    // --------------------------------------------------------
    // [중요] Ref 데이터 -> HLS 데이터로 복사 (형변환)
    // --------------------------------------------------------
    printf("Converting data types for HLS...\n");
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < dk; k++) {
            Q_hls[i][k] = Q_ref[i][k]; 
            K_hls[i][k] = K_ref[i][k];
        }
        for (int v = 0; v < dv; v++) {
            V_hls[i][v] = V_ref[i][v];
            Output_HLS[i][v] = 0;
        }
    }

    // --------------------------------------------------------
    // Reference 계산 (FP32)
    // --------------------------------------------------------
    printf("Computing reference attention (FP32)...\n");
    reference_attention_fp32(Q_ref, K_ref, V_ref, Q_scale, K_scale, V_scale, Output_ref);

    // --------------------------------------------------------
    // HLS 커널 호출 (수정됨: 인자 순서 변경)
    // --------------------------------------------------------
    printf("Running HLS kernel...\n");
    
    // [수정 완료] Output_HLS를 4번째 인자로 넣었습니다.
    // 기존: (Q, K, V, Q_scale, ..., Output) -> 에러
    // 수정: (Q, K, V, Output, Q_scale, ...) -> 정상
    compute_attention_HLS(Q_hls, K_hls, V_hls, Output_HLS, Q_scale, K_scale, V_scale);

    // --------------------------------------------------------
    // 결과 비교
    // --------------------------------------------------------
    printf("\nComparing results...\n");
    
    double mse = 0.0;
    double max_error = 0.0;
    int max_error_i = 0, max_error_d = 0;
    
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dv; d++) {
            // HLS 결과(fixed_t)를 float으로 변환하여 비교
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