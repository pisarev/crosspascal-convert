// Tensor decoding routines derived from ggml-org/llama.cpp.
// Pinned source and MIT license are recorded in PIN.json and LICENSE.
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#define __extension__
#include "ggml-common.h"
template<typename T> static __device__ __forceinline__ T ggml_cuda_cast(float x) {
    return static_cast<T>(x);
}
template<typename dst_t>
static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = tid/4;
    const int64_t t = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(tid%4);
    const int64_t n = t / 4;
    const int64_t j = t - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is+0] >> 4) & 3) << 4) :
                          (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[ib].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + 128*n + 32*j;
    const uint8_t * q = x[ib].qs + 32*n;
    const uint8_t * hm = x[ib].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = ggml_cuda_cast<dst_t>(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;

    // assume 32 threads
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + 64*il + n*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * q = x[ib].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = ggml_cuda_cast<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l +32] = ggml_cuda_cast<dst_t>(d2 * (q[l] >>  4) - m2);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + 64*il + 2*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + 128*ip + il;

    const float d = x[ib].d;

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t   qh = x[ib].qh[32*ip + il];
    const int8_t  * sc = x[ib].scales + is;

    y[ 0] = ggml_cuda_cast<dst_t>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = ggml_cuda_cast<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = ggml_cuda_cast<dst_t>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = ggml_cuda_cast<dst_t>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t * qs = x[ibs].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[ibs].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[ibs].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[ibs].d * (1 + 2*((x[ibs].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq4_nl * x = (const block_iq4_nl *) vx + ibs*(QK_K/QK4_NL);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_xs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ibs].qs + 16*ib + 4*il;
    const float d = (float)x[ibs].d * ((((x[ibs].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[ibs].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}
extern "C" __global__ void cp_dequant_q3_K(const void *source, float *dest, long long blocks) {
    const long long index = blockIdx.x;
    if (index >= blocks) return;
    dequantize_q3_K(source, index, dest + index*256, threadIdx.x);
}

extern "C" __global__ void cp_dequant_q4_K(const void *source, float *dest, long long blocks) {
    const long long index = blockIdx.x;
    if (index >= blocks) return;
    dequantize_q4_K(source, index, dest + index*256, threadIdx.x);
}

extern "C" __global__ void cp_dequant_q5_K(const void *source, float *dest, long long blocks) {
    const long long index = blockIdx.x;
    if (index >= blocks) return;
    dequantize_q5_K(source, index, dest + index*256, threadIdx.x);
}

extern "C" __global__ void cp_dequant_q6_K(const void *source, float *dest, long long blocks) {
    const long long index = blockIdx.x;
    if (index >= blocks) return;
    dequantize_q6_K(source, index, dest + index*256, threadIdx.x);
}

extern "C" __global__ void cp_dequant_iq3_s(const void *source, float *dest, long long blocks) {
    const long long index = blockIdx.x;
    if (index >= blocks) return;
    dequantize_iq3_s(source, index, dest + index*256, threadIdx.x);
}

extern "C" __global__ void cp_dequant_iq4_xs(const void *source, float *dest, long long blocks) {
    const long long index = blockIdx.x;
    if (index >= blocks) return;
    dequantize_iq4_xs(source, index, dest + index*256, threadIdx.x);
}

extern "C" __global__ void cp_dequant_iq4_nl(const void *source, float *dest, long long blocks) {
    const long long group = blockIdx.x;
    if (group*8 + threadIdx.x%8 >= blocks) return;
    dequantize_iq4_nl(source, group, dest + group*256, threadIdx.x);
}
extern "C" __global__ void cp_dequant_q8_0(const void *source, float *dest, long long blocks) {
    const long long index = blockIdx.x;
    if (index >= blocks) return;
    const block_q8_0 *x = static_cast<const block_q8_0 *>(source);
    dest[index*32 + threadIdx.x] = static_cast<float>(x[index].d)*x[index].qs[threadIdx.x];
}
extern "C" __global__ void cp_dequant_f32(const float *source, float *dest, long long count) {
    const long long index = static_cast<long long>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index < count) dest[index] = source[index];
}
extern "C" __global__ void cp_dequant_bf16(const unsigned short *source, float *dest, long long count) {
    const long long index = static_cast<long long>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index < count) dest[index] = __uint_as_float(static_cast<unsigned>(source[index]) << 16);
}
