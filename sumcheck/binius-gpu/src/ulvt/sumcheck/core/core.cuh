#pragma once
#include <cstdint>

#include "../utils/constants.hpp"

#ifndef CUDA_CHECK
#define CUDA_CHECK
#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

__global__ void calculate_multilinear_product_sums_kernel( // can possibly tile becuase a lot of data reuse
	const uint32_t* multilinear_evaluations_p1_unbitsliced, // eq polynomial in zerocheck (F(2^128))
	const uint32_t* multilinear_evaluations_p1, // eq polynomial in zerocheck (F(2^128))
	const uint32_t* multilinear_evaluations, // binary
	uint32_t* destination,
	const uint32_t d,
	const uint32_t round_idx,
	const uint32_t n
);

__host__ __device__ void calculate_interpolation_points(
	const uint32_t* multilinear_evaluations_p1_unbitsliced, // eq polynomial in zerocheck (F(2^128))
	//const uint32_t* multilinear_evaluations_p1, // eq polynomial in zerocheck (F(2^128))
	const uint32_t* multilinear_evaluations, // binary
	const uint32_t* random_challenges,
	uint32_t* destination,
	uint32_t* claimed_sum,
	const uint32_t d,
	const uint32_t round_idx,
	const uint32_t n
);

__host__ __device__ void calculate_random_challenge_products(
	const uint32_t* random_challenges,
	uint32_t* destination,
	const uint32_t d,
	const uint32_t round_idx
);

__host__ __device__ void calculate_interpolation_point_products(
	const uint32_t interpolation_point,
	uint32_t* destination,
	const uint32_t d,
	const uint32_t round_idx	
);

__host__ __device__ void evaluate_composition_on_batch_row(
	const uint32_t* first_batch_of_row,
	uint32_t* batch_composition_destination,
	const uint32_t composition_size,
	const uint32_t original_evals_per_col
);


__host__ __device__ void evaluate_composition_on_batch_row_gpu(
	const uint32_t* first_batch_of_row,
	uint32_t* batch_composition_destination,
	const uint32_t composition_size,
	const uint32_t original_evals_per_col
);

__host__ __device__ void fold_batch(
	const uint32_t lower_batch[BITS_WIDTH],
	const uint32_t upper_batch[BITS_WIDTH],
	uint32_t dst_batch[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const bool is_interpolation
);

void fold_small(
	const uint32_t source[BITS_WIDTH],
	uint32_t destination[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const uint32_t list_len
);

__host__ __device__ void compute_sum(
	uint32_t sum[INTS_PER_VALUE],
	uint32_t bitsliced_batch[BITS_WIDTH],
	const uint32_t num_eval_points_being_summed_unpadded
);

__host__ void fold_multiple(
	const uint32_t* folded_mutlilinear_evaluations_p1,
	const uint32_t* multilinear_evaluations_d,
	const uint32_t* random_challenges,
	uint32_t* destination,
	const uint32_t round_idx,
	const uint32_t n,
	const uint32_t d
);

__host__ __device__ void get_batch(
	const uint32_t* multilinear_evaluations_d,
	const uint32_t* random_challenges_subset_products, 
	uint32_t* destination,
	const uint32_t idx,
	const uint32_t p_idx,
	const uint32_t round_idx,
	const uint32_t n
);

__host__ void calculate_random_challenge_subset_products(const uint32_t* random_challenges, uint32_t* destination, const uint32_t round_idx);