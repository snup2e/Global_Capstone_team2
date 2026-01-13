#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>



#define Br 32 
#define Bc 32 
typedef ap_int<8> qint8_t;
typedef ap_int<32> qint32_t;

typedef ap_fixed<24, 8> scale_fixed_t;

// 출력용 fixed point 타입
typedef ap_fixed<16, 5> fixed_t;
// 중간 계산용 타입
typedef ap_fixed<32, 16> calc_t;

#define N   512     
#define dk  64     
#define dv  64      


void compute_attention_HLS(
    qint8_t Q[N][dk],           
    qint8_t K[N][dk],           
    qint8_t V[N][dv],           
    fixed_t Output[N][dv],     
    float scale_Q[N],         
    float scale_K[N],          
    float scale_V[N]           
);
