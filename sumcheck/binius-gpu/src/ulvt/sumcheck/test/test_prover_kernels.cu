#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <cstdlib>
#include "../core/kernels.cuh"
#include "../core/core.cuh"
#include "utils/unbitsliced_utils.cuh"
//#include "utils/tower_7_mul.cuh"
#include "utils/unbitsliced_mul.cuh"

__host__ __device__ __uint128_t tower_height_7_mul(const __uint128_t a, const __uint128_t b) {
	uint64_t a1 = (uint64_t)(a >> 64);
	uint64_t a0 = (uint64_t)(a & ((__uint128_t)0xffffffffffffffff));

	uint64_t b1 = (uint64_t)(b >> 64);
	uint64_t b0 = (uint64_t)(b & ((__uint128_t)0xffffffffffffffff));

	uint64_t a0b0 = FanPaarTowerField<6>::multiply(a0, b0);
	uint64_t a0b1 = FanPaarTowerField<6>::multiply(a0, b1);
	uint64_t a1b0 = FanPaarTowerField<6>::multiply(a1, b0);
	uint64_t a1b1 = FanPaarTowerField<6>::multiply(a1, b1);

	uint64_t result_bottom_half = a0b0 ^ a1b1;
	uint64_t result_top_half = a0b1 ^ a1b0 ^ FanPaarTowerField<6>::multiply_alpha(a1b1);

	return ((__uint128_t)result_top_half) << 64 | ((__uint128_t)result_bottom_half);
}