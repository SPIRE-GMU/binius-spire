#pragma once

#include <cstdint>

#include "../utils/constants.hpp"
#include "../../finite_fields/circuit_generator/unrolled/binary_tower_rolled.cuh"
#include "../../finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "core.cuh"

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

template<uint32_t COMPOSITION_SIZE, uint32_t EVALS_PER_MULTILINEAR>
__global__ void bitpack_kernel(const uint32_t* evals, uint32_t* destination) {
	uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t unpacked_bit_idx = idx * INTS_PER_VALUE;

	// printf("composition_size %u\n", COMPOSITION_SIZE);
	//printf("idx %u out of %u\n", idx, COMPOSITION_SIZE * EVALS_PER_MULTILINEAR);
	//if(unpacked_bit_idx < (uint64_t) COMPOSITION_SIZE * EVALS_PER_MULTILINEAR*INTS_PER_VALUE) {
	if(idx < COMPOSITION_SIZE * EVALS_PER_MULTILINEAR) {
		if(idx / 32 < COMPOSITION_SIZE * EVALS_PER_MULTILINEAR / 32) {
			if(idx % 32 == 0) destination[idx / 32] = 0;
			//if(idx == 0) destination[0] = 0;
			__syncthreads();
			atomicOr(destination + idx / 32, evals[unpacked_bit_idx] << (idx % 32));
		} else {
			printf("HERE\n");
		}
	}
}


template <uint32_t INTERPOLATION_POINTS, uint32_t COMPOSITION_SIZE, uint32_t EVALS_PER_MULTILINEAR>
__global__ void compute_compositions( // evaluates Si(Xi) at multiple points and gets the claimed sum
	const uint32_t* multilinear_evaluations, // d x 2^n table representing the 3 multiplied hypercubes
	uint32_t* multilinear_products_sums, 
	uint32_t* folded_products_sums,
	const uint32_t coefficients[INTERPOLATION_POINTS * BITS_WIDTH],
	const uint32_t num_batch_rows,
	const uint32_t active_threads,
	const uint32_t active_threads_folded
) {
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;  // start the batch index off at the tid

	uint32_t folded_products_sums_this_thread[INTERPOLATION_POINTS * BITS_WIDTH];
	uint32_t multilinear_products_sums_this_thread[BITS_WIDTH];

	memset(folded_products_sums_this_thread, 0, INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t));
	memset(multilinear_products_sums_this_thread, 0, BITS_WIDTH * sizeof(uint32_t));

	for (uint64_t row_idx = tid; row_idx < num_batch_rows; row_idx += gridDim.x * blockDim.x) {
		uint32_t this_multilinear_product[BITS_WIDTH];

		// finding the claimed sum P(000) + P(001) + P(010) + ... + P(110) + P(111)
		evaluate_composition_on_batch_row( 
			multilinear_evaluations + BITS_WIDTH * row_idx, // the row_idx'th batch 
			this_multilinear_product, // destination for p1p2p3...pd
			COMPOSITION_SIZE, // =d
			EVALS_PER_MULTILINEAR // number of elements in multilinear (2^n) to define the striding of composition
		);

		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			multilinear_products_sums_this_thread[i] ^= this_multilinear_product[i]; // add to running batch sums
		}

		uint32_t num_batch_rows_to_fold = num_batch_rows / 2;

		if (row_idx < num_batch_rows_to_fold) {
			uint32_t folded_batch_row[INTERPOLATION_POINTS * COMPOSITION_SIZE * BITS_WIDTH];

			for (int column_idx = 0; column_idx < COMPOSITION_SIZE; ++column_idx) {
				uint32_t batches_fitting_into_original_column = EVALS_PER_MULTILINEAR / 32;
				const uint32_t* lower_batch =
					multilinear_evaluations +
					BITS_WIDTH * (batches_fitting_into_original_column * column_idx + row_idx);
				const uint32_t* upper_batch = lower_batch + BITS_WIDTH * num_batch_rows_to_fold;
				for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
					fold_batch( // 3%
						lower_batch,
						upper_batch,
						folded_batch_row + BITS_WIDTH * (column_idx * INTERPOLATION_POINTS + interpolation_point),
						coefficients + BITS_WIDTH * interpolation_point, 
						true
					);
				}
			}

			uint32_t this_interpolation_point_product_batch[BITS_WIDTH];
			for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
				evaluate_composition_on_batch_row( // THIS IS THE BIGGEST SLOWDOWN
					folded_batch_row + BITS_WIDTH * interpolation_point, // starting batch
					this_interpolation_point_product_batch, // destination 
					COMPOSITION_SIZE, // number of batches to multiply (number of multilinear polynomials together)
					INTERPOLATION_POINTS * 32 // stride to let the algorithm determine whcih batches to multiply
				);
				uint32_t* this_interpolation_point_sum_location =
					folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;
				for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
					this_interpolation_point_sum_location[i] ^= this_interpolation_point_product_batch[i];
				}
			}
		}
	}

	if (tid < active_threads) {
		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			atomicXor(multilinear_products_sums + i, multilinear_products_sums_this_thread[i]); // TODO instead of atomic xor may speedup by a few %
		}
	}

	if (tid < active_threads_folded) {
		for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
			uint32_t* batch_to_copy_to = folded_products_sums + BITS_WIDTH * interpolation_point;
			uint32_t* batch_to_copy_from = folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;

			for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
				atomicXor(batch_to_copy_to + i, batch_to_copy_from[i]);
			}
		}
	}
}

template <uint32_t INTERPOLATION_POINTS, uint32_t COMPOSITION_SIZE, uint32_t EVALS_PER_MULTILINEAR>
__global__ void compute_compositions_using_get_batch( // evaluates Si(Xi) at multiple points and gets the claimed sum
	const uint32_t* multilinear_evaluations_p1,
	const uint32_t* multilinear_evaluations, // d x 2^n table representing the 3 multiplied hypercubes
	const uint32_t* random_challenges_subset_products,
	uint32_t* multilinear_products_sums, 
	uint32_t* folded_products_sums,
	const uint32_t coefficients[INTERPOLATION_POINTS * BITS_WIDTH],
	const uint32_t num_batch_rows,
	const uint32_t active_threads,
	const uint32_t active_threads_folded,
	const uint32_t round_idx,
	const uint32_t n
) {
	// __shared__ uint32_t random_challenges_subset_products_s[(1 << 7) * BITS_WIDTH];

	// for(int i = threadIdx.x; i < (1 << round_idx) * BITS_WIDTH; i += blockIdx.x) {
	// 	printf("set %u\n", i);
	// 	random_challenges_subset_products_s[i] = random_challenges_subset_products[i];
	// }

	// __syncthreads();


	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;  // start the batch index off at the tid

	uint32_t folded_products_sums_this_thread[INTERPOLATION_POINTS * BITS_WIDTH];
	uint32_t multilinear_products_sums_this_thread[BITS_WIDTH];

	memset(folded_products_sums_this_thread, 0, INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t));
	memset(multilinear_products_sums_this_thread, 0, BITS_WIDTH * sizeof(uint32_t));

	for (uint32_t row_idx = tid; row_idx < num_batch_rows; row_idx += gridDim.x * blockDim.x) {
		uint32_t this_multilinear_product[BITS_WIDTH];

		memcpy(this_multilinear_product, multilinear_evaluations_p1 + BITS_WIDTH * row_idx, BITS_WIDTH * sizeof(uint32_t));
		for(int i = 1; i < COMPOSITION_SIZE; i++) {
			uint32_t batch[BITS_WIDTH];
			memset(batch, 0, BITS_WIDTH*sizeof(uint32_t));
			get_batch(multilinear_evaluations, random_challenges_subset_products, batch, row_idx, i-1, round_idx, n);
			multiply_unrolled<TOWER_HEIGHT>(this_multilinear_product, batch, this_multilinear_product);
		}

		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			multilinear_products_sums_this_thread[i] ^= this_multilinear_product[i]; // add to running batch sums
		}

		uint32_t num_batch_rows_to_fold = num_batch_rows / 2;

		if (row_idx < num_batch_rows_to_fold) {
			uint32_t folded_batch_row[INTERPOLATION_POINTS * COMPOSITION_SIZE * BITS_WIDTH];

			for (int column_idx = 0; column_idx < COMPOSITION_SIZE; ++column_idx) {
				uint32_t batches_fitting_into_original_column = EVALS_PER_MULTILINEAR / 32;
				if(column_idx < COMPOSITION_SIZE-1) {
					uint32_t lower_batch[BITS_WIDTH], upper_batch[BITS_WIDTH];
					get_batch(multilinear_evaluations, random_challenges_subset_products, lower_batch, row_idx, column_idx, round_idx, n);
					get_batch(multilinear_evaluations, random_challenges_subset_products, upper_batch, row_idx + num_batch_rows_to_fold, column_idx, round_idx, n);
					for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point)
					{
						fold_batch( // 3%
							lower_batch,
							upper_batch,
							folded_batch_row + BITS_WIDTH * (column_idx * INTERPOLATION_POINTS + interpolation_point),
							coefficients + BITS_WIDTH * interpolation_point,
							true);
					}
				} else {
					const uint32_t *lower_batch, *upper_batch;
					lower_batch = multilinear_evaluations_p1 +
					BITS_WIDTH * row_idx;
					upper_batch = lower_batch + BITS_WIDTH * num_batch_rows_to_fold;
					for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point)
					{
						fold_batch( // 3%
							lower_batch,
							upper_batch,
							folded_batch_row + BITS_WIDTH * (column_idx * INTERPOLATION_POINTS + interpolation_point),
							coefficients + BITS_WIDTH * interpolation_point,
							true);
					}
				}
			}

			uint32_t this_interpolation_point_product_batch[BITS_WIDTH];
			for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
				evaluate_composition_on_batch_row( // THIS IS THE BIGGEST SLOWDOWN
					folded_batch_row + BITS_WIDTH * interpolation_point, // starting batch
					this_interpolation_point_product_batch, // destination 
					COMPOSITION_SIZE, // number of batches to multiply (number of multilinear polynomials together)
					INTERPOLATION_POINTS * 32 // stride to let the algorithm determine whcih batches to multiply
				);
				uint32_t* this_interpolation_point_sum_location =
					folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;
				for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
					this_interpolation_point_sum_location[i] ^= this_interpolation_point_product_batch[i];
				}
			}
		}
	}

	if (tid < active_threads) {
		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			atomicXor(multilinear_products_sums + i, multilinear_products_sums_this_thread[i]); // TODO instead of atomic xor may speedup by a few %
		}
	}

	if (tid < active_threads_folded) {
		for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
			uint32_t* batch_to_copy_to = folded_products_sums + BITS_WIDTH * interpolation_point;
			uint32_t* batch_to_copy_from = folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;

			for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
				atomicXor(batch_to_copy_to + i, batch_to_copy_from[i]);
			}
		}
	}
}




__global__ void fold_large_list_halves(
	uint32_t* source,
	uint32_t* destination,
	uint32_t coefficient[BITS_WIDTH],
	const uint32_t num_batch_rows,
	const uint32_t src_evals_per_column,
	const uint32_t dst_evals_per_column,
	const uint32_t num_cols
);


__global__ void print_debug(uint32_t* source, const uint32_t num_batch_rows);