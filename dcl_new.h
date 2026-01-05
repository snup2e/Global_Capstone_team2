
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>

typedef ap_fixed<16, 5> fixed_t;

// Define tensor dimensions
#define N   32     // Sequence length
#define dk  64     // Key/Query dimension
#define dv  64     // Value dimension

void compute_attention_HLS(fixed_t Q[N][dk], fixed_t K[N][dk], fixed_t V[N][dv], fixed_t Output[N][dv]);
