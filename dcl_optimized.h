#ifndef DCL_OPTIMIZED_H
#define DCL_OPTIMIZED_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>

// 출력용 fixed point 타입
typedef ap_fixed<16, 5> fixed_t;

// 중간 계산용 타입
typedef ap_fixed<32, 16> calc_t;

// Define tensor dimensions
#define N   512     // Sequence length
#define dk  64      // Key/Query dimension
#define dv  64      // Value dimension

// --------------------------------------------------------
// INT8 양자화 구조
// - Q, K, V: int8_t [N][dim]
// - 각 행(row)마다 scale factor 존재: float [N]
// - 실제값 = int8_value * scale_factor
// --------------------------------------------------------

void compute_attention_HLS(
    int8_t Q[N][dk],
    int8_t K[N][dk],
    int8_t V[N][dv],
    float Q_scale[N],      // Q의 행별 스케일 팩터
    float K_scale[N],      // K의 행별 스케일 팩터
    float V_scale[N],      // V의 행별 스케일 팩터
    fixed_t Output[N][dv]
);

#endif // DCL_OPTIMIZED_H
