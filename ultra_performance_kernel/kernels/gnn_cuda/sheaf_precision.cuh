// ==============================================================================
// SHEAF MULTI-PRECISION SYSTEM: FP32, FP16, BF16, FP8, FP4
// Architectural Target: NVIDIA Turing (SM_75) & Modern SIMT Accelerators
// Purpose: Provide high-throughput tensor storage and fast register-level
//          dequantization across half-precision, brain-float, and micro-float types.
// ==============================================================================

#ifndef SHEAF_PRECISION_CUH
#define SHEAF_PRECISION_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if __CUDA_ARCH__ >= 800 || (defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 11)
#include <cuda_bf16.h>
#endif
#include <stdint.h>

/// Supported physical execution precisions for sheaf diffusion
enum SheafPrecisionMode {
    PRECISION_FP32 = 0,  // Full precision 32-bit IEEE float (Baseline)
    PRECISION_FP16 = 1,  // Half precision 16-bit float (Native SM_75 dual ALU)
    PRECISION_BF16 = 2,  // Brain float 16-bit (8-bit exponent, 7-bit mantissa)
    PRECISION_FP8  = 3,  // 8-bit Micro-float (E4M3 format, 4x memory compression)
    PRECISION_FP4  = 4   // 4-bit Sub-byte float (E2M1 format, 8x memory compression)
};

// ==============================================================================
// FP8 (E4M3) FAST REGISTER CONVERSION
// Sign: 1 bit [7], Exponent: 4 bits [6:3], Mantissa: 3 bits [2:0], Bias: 7
// ==============================================================================
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t byte) {
    if (byte == 0x00) return 0.0f;
    if (byte == 0x80) return -0.0f;
    
    uint32_t sign = (byte & 0x80) << 24;
    int exp = (byte >> 3) & 0x0F;
    uint32_t mant = (byte & 0x07);

    if (exp == 0) {
        // Subnormal: (-1)^sign * 2^(-6) * (mant / 8)
        float val = (float)mant / 8.0f * (1.0f / 64.0f);
        return (byte & 0x80) ? -val : val;
    } else {
        // Normal: (-1)^sign * 2^(exp - 7) * (1 + mant / 8)
        uint32_t f32_exp = (exp - 7 + 127) << 23;
        uint32_t f32_mant = mant << 20;
        uint32_t f32_bits = sign | f32_exp | f32_mant;
        return __uint_as_float(f32_bits);
    }
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val) {
    if (val == 0.0f) return 0x00;
    uint32_t bits = __float_as_uint(val);
    uint8_t sign = (bits >> 24) & 0x80;
    int exp = ((bits >> 23) & 0xFF) - 127 + 7;
    uint32_t mant = (bits >> 20) & 0x07;

    if (exp <= 0) {
        return sign; // flush small values to zero
    }
    if (exp >= 15) {
        return sign | 0x7E; // clamp to max representable normal
    }
    return sign | ((exp & 0x0F) << 3) | (mant & 0x07);
}

// ==============================================================================
// FP4 (E2M1) FAST REGISTER CONVERSION
// Packed: 2 values per byte. High nibble [7:4] = elem 0, Low nibble [3:0] = elem 1.
// Format: Sign [3], Exponent [2:1], Mantissa [0].
// Representable magnitudes: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// ==============================================================================
__device__ __forceinline__ float fp4_nibble_to_float(uint8_t nibble) {
    const float FP4_LUT[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float mag = FP4_LUT[nibble & 0x07];
    return (nibble & 0x08) ? -mag : mag;
}

__device__ __forceinline__ void decode_fp4_pair(uint8_t byte, float& out0, float& out1) {
    out0 = fp4_nibble_to_float((byte >> 4) & 0x0F);
    out1 = fp4_nibble_to_float(byte & 0x0F);
}

__device__ __forceinline__ uint8_t float_to_fp4_nibble(float val) {
    uint8_t sign = (val < 0.0f) ? 0x08 : 0x00;
    float abs_val = fabsf(val);
    uint8_t code = 0;
    if (abs_val < 0.25f) code = 0;
    else if (abs_val < 0.75f) code = 1;
    else if (abs_val < 1.25f) code = 2;
    else if (abs_val < 1.75f) code = 3;
    else if (abs_val < 2.5f) code = 4;
    else if (abs_val < 3.5f) code = 5;
    else if (abs_val < 5.0f) code = 6;
    else code = 7;
    return sign | code;
}

// ==============================================================================
// BF16 AND FP16 CONVERSION HELPERS
// ==============================================================================
__device__ __forceinline__ float bfloat16_to_float(uint16_t bf_bits) {
    uint32_t f_bits = ((uint32_t)bf_bits) << 16;
    return __uint_as_float(f_bits);
}

__device__ __forceinline__ uint16_t float_to_bfloat16(float f) {
    uint32_t f_bits = __float_as_uint(f);
    // Round to nearest even truncation
    uint32_t rounding = 0x7FFF + ((f_bits >> 16) & 1);
    return (uint16_t)((f_bits + rounding) >> 16);
}

__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

#endif // SHEAF_PRECISION_CUH
