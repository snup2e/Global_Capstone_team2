#include "dcl_new.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib> // for rand

using namespace std;

// 한 행(Row)단위로 양자화를 수행하는 함수
void quantize_row_wise(
    const vector<float>& src, 
    vector<int8_t>& dst_data, 
    vector<float>& dst_scales, 
    int rows, 
    int cols
) {
    for (int i = 0; i < rows; ++i) {
        // 1. 해당 행의 절대값 최댓값 찾기
        float max_val = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float val = std::abs(src[i * cols + j]);
            if (val > max_val) max_val = val;
        }

        // 2. Scale 계산 (int8 범위: -127 ~ 127)
        // max_val이 0이면 scale을 1.0으로 설정하여 0 나누기 방지
        float scale = (max_val > 0) ? (max_val / 127.0f) : 1.0f;
        dst_scales[i] = scale;

        // 3. 양자화 및 저장
        for (int j = 0; j < cols; ++j) {
            float float_val = src[i * cols + j];
            float quantized = float_val / scale;

            // 반올림 및 클리핑 (-127 ~ 127)
            int val_int = (int)round(quantized);
            if (val_int > 127) val_int = 127;
            if (val_int < -127) val_int = -127;

            dst_data[i * cols + j] = (int8_t)val_int;
        }
    }
}

void generate_attention_matrices() {
    // 파일 열기 (데이터용, 스케일용)
    ofstream Q_file("Q_int8.bin", ios::binary);
    ofstream K_file("K_int8.bin", ios::binary);
    ofstream V_file("V_int8.bin", ios::binary);
    
    ofstream Q_scale_file("Q_scales.bin", ios::binary);
    ofstream K_scale_file("K_scales.bin", ios::binary);
    ofstream V_scale_file("V_scales.bin", ios::binary);

    if (!Q_file || !Q_scale_file) {
        cerr << "Error opening file for writing." << endl;
        exit(1);
    }

    // 임시 Float 버퍼 (랜덤 생성용)
    vector<float> Q_float(N * dk);
    vector<float> K_float(N * dk);
    vector<float> V_float(N * dv);

    // Int8 데이터 및 Scale 저장 버퍼
    vector<int8_t> Q_int8(N * dk), K_int8(N * dk), V_int8(N * dv);
    vector<float> Q_scales(N), K_scales(N), V_scales(N);

    srand(42); 

    // 랜덤 데이터 생성 (-1.0 ~ 1.0)
    for (int i = 0; i < N * dk; ++i) {
        Q_float[i] = static_cast<float>(rand()) / RAND_MAX * 2 - 1;
        K_float[i] = static_cast<float>(rand()) / RAND_MAX * 2 - 1;
    }
    for (int i = 0; i < N * dv; ++i) {
        V_float[i] = static_cast<float>(rand()) / RAND_MAX * 2 - 1;
    }

    // Per-Row Quantization 수행
    quantize_row_wise(Q_float, Q_int8, Q_scales, N, dk);
    quantize_row_wise(K_float, K_int8, K_scales, N, dk);
    quantize_row_wise(V_float, V_int8, V_scales, N, dv);

    // 바이너리 쓰기 (데이터)
    Q_file.write(reinterpret_cast<char*>(Q_int8.data()), N * dk * sizeof(int8_t));
    K_file.write(reinterpret_cast<char*>(K_int8.data()), N * dk * sizeof(int8_t));
    V_file.write(reinterpret_cast<char*>(V_int8.data()), N * dv * sizeof(int8_t));

    // 바이너리 쓰기 (스케일 - 각 행마다 하나씩, 총 N개)
    Q_scale_file.write(reinterpret_cast<char*>(Q_scales.data()), N * sizeof(float));
    K_scale_file.write(reinterpret_cast<char*>(K_scales.data()), N * sizeof(float));
    V_scale_file.write(reinterpret_cast<char*>(V_scales.data()), N * sizeof(float));

    cout << "Generated Per-Row Quantized tensors." << endl;
    cout << "Data Files: *_int8.bin (" << N << "x" << dk << "/" << dv << " bytes)" << endl;
    cout << "Scale Files: *_scales.bin (" << N << " floats)" << endl;
}

int main() {
    generate_attention_matrices();
    return 0;
}