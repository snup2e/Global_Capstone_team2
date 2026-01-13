#include "dcl_new.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

// Softmax (Float 연산)
void softmax(float matrix[N][N]) {
    for (int i = 0; i < N; ++i) {
        float max_val = matrix[i][0];
        for (int j = 1; j < N; ++j) {
            if (matrix[i][j] > max_val) max_val = matrix[i][j];
        }

        float sum = 0;
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = exp(matrix[i][j] - max_val);
            sum += matrix[i][j];
        }

        for (int j = 0; j < N; ++j) {
            matrix[i][j] /= sum;
        }
    }
}

// Per-Row Scale을 적용한 Attention 계산
void compute_attention(
    int8_t* Q, float* Q_scales,
    int8_t* K, float* K_scales,
    int8_t* V, float* V_scales,
    float* Output
) {
    // Attention Score 행렬 (Float)
    // 스택 오버플로우 주의: N이 크면 vector<vector<float>> 사용 권장
    static float attention[N][N]; 
    float dk_inv_sqrt = 1.0f / sqrt((float)dk);

    // 1. Compute Q * K^T with Dequantization
    for (int i = 0; i < N; ++i) {       // Q의 행 인덱스
        for (int j = 0; j < N; ++j) {   // K의 행(Transposed 열) 인덱스
            
            // Int32 Accumulation (FPGA의 DSP 연산 모사)
            int32_t int_sum = 0;
            for (int k = 0; k < dk; ++k) {
                int_sum += (int32_t)Q[i * dk + k] * (int32_t)K[j * dk + k];
            }

            // Dequantize: (IntSum) * Scale_Q[i] * Scale_K[j] * 1/sqrt(dk)
            // 여기서 i는 Query의 행, j는 Key의 행(논리적 Key)
            float total_scale = Q_scales[i] * K_scales[j] * dk_inv_sqrt;
            attention[i][j] = (float)int_sum * total_scale;
        }
    }

    // 2. Apply Softmax
    softmax(attention);

    // 3. Compute Attention * V with Dequantization
    for (int i = 0; i < N; ++i) {       // Output의 행
        for (int j = 0; j < dv; ++j) {  // Output의 열 (Value의 차원)
            
            float out_sum = 0.0f;
            for (int k = 0; k < N; ++k) { // Attention 확률과 Value 행렬 곱
                
                // V Dequantize: V[k][j]는 Value의 k번째 행 데이터
                // 따라서 Scale도 V_scales[k]를 사용해야 함
                float v_val_real = (float)V[k * dv + j] * V_scales[k];
                
                out_sum += attention[i][k] * v_val_real;
            }
            Output[i * dv + j] = out_sum;
        }
    }
}

// 파일 로드 헬퍼 함수
template <typename T>
void load_bin(const char* filename, T* buffer, int count) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(buffer), count * sizeof(T));
    file.close();
}

int main() {
    // 메모리 할당 (Vector 사용 권장)
    vector<int8_t> Q(N * dk), K(N * dk), V(N * dv);
    vector<float> Q_scales(N), K_scales(N), V_scales(N);
    vector<float> Output(N * dv);

    // 데이터 로드 (Int8)
    load_bin("Q_int8.bin", Q.data(), N * dk);
    load_bin("K_int8.bin", K.data(), N * dk);
    load_bin("V_int8.bin", V.data(), N * dv);

    // 스케일 로드 (Float, N개)
    load_bin("Q_scales.bin", Q_scales.data(), N);
    load_bin("K_scales.bin", K_scales.data(), N);
    load_bin("V_scales.bin", V_scales.data(), N);

    cout << "Loaded quantized data and scales." << endl;

    // Attention 계산
    compute_attention(
        Q.data(), Q_scales.data(),
        K.data(), K_scales.data(),
        V.data(), V_scales.data(),
        Output.data()
    );

    // 결과 저장 (검증을 위해 Float 그대로 저장)
    ofstream outfile("Output_tensor.bin", ios::binary);
    outfile.write(reinterpret_cast<char*>(Output.data()), N * dv * sizeof(float));
    outfile.close();

    cout << "Computation complete. Saved Output_tensor.bin (Float32)" << endl;

    return 0;
}