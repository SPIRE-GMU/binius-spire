#pragma once

#include <cstdint>

#include "../utils/constants.hpp"
#include "../../finite_fields/circuit_generator/unrolled/binary_tower_rolled.cuh"
#include "../../finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"

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

__global__ void multiply_hybrid_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) {
    multiply_thread(field_element_a, field_element_b, destination, num_bits, threadIdx.x, blockIdx.x);
}

__global__ void multiply_then_add_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) {
    __shared__ uint32_t batch_product[BITS_WIDTH];
    multiply_thread(field_element_a, field_element_b, batch_product, num_bits, threadIdx.x, blockIdx.x);
	__syncthreads();
    atomicXor(destination + threadIdx.x, batch_product[threadIdx.x]);
}

__global__ void composition_then_add_kernel(const uint32_t* field_elements, uint32_t* destination, uint32_t num_bits, uint32_t composition_size) {
    uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
    uint32_t num_elements = gridDim.x;

	if(bid == 0) {
		destination[tid] = 0;
	}
	
    __shared__ uint32_t batch_product[BITS_WIDTH];
	batch_product[tid] = 0;
	__syncthreads();
	batch_product[tid] = field_elements[bid * num_bits + tid];
	__syncthreads();

	//printf("%d\n", field_elements[bid * num_bits + tid]);
	if(tid == 0 && bid == 0) {
		printf("multilinear_evaluations[0] = %u\n", field_elements[0]);
	}

    for(int i = 1; i < composition_size; i++) {
		//printf("multiply_thread field_eleemnts + %d, num_bits=%d, tid=%d\n", i*num_elements*num_bits + bid*num_bits, num_bits, tid);
        multiply_thread(batch_product, field_elements + i*num_elements*num_bits + bid*num_bits, batch_product, num_bits, tid, 0);
		__syncthreads();
    }

    atomicXor(destination + tid, batch_product[tid]);
}

__global__ void interpolation_then_composition_then_add(const uint32_t* batches, const uint32_t* coefficient, uint32_t* destination, uint32_t num_bits, uint32_t num_batches, uint32_t composition_size) { // calculate si(xi)
    uint32_t tid = threadIdx.x;
	uint32_t idx = tid + blockIdx.x * blockDim.x;// 127 + 128 * num_batch_rows / 2
    
    __shared__ uint32_t xor_of_halves[BITS_WIDTH];
	__shared__ uint32_t folded_points[BITS_WIDTH];
	__shared__ uint32_t composition[BITS_WIDTH];

	composition[tid] = 0;
	xor_of_halves[tid] = 0;
	folded_points[tid] = 0;

	if(blockIdx.x == 0) {
		destination[tid] = 0;
	}

	__syncthreads();
	//composition[tid] = 0xFFFFFFFF;

	for(int j = 0; j < composition_size; j++) {
		/*if(idx == 0) {
			printf("fine coef[0] = %u\n", coefficient[0]);
		}*/
		const uint32_t* lower_batches = batches + j * num_batches * BITS_WIDTH;
		
		//xor_of_halves[tid] = lower_batches[idx] ^ lower_batches[idx + num_batches * BITS_WIDTH / 2];
		xor_of_halves[tid] = batches[idx + j*num_batches*BITS_WIDTH] ^ batches[idx + j*num_batches*BITS_WIDTH + num_batches*BITS_WIDTH/2];
		folded_points[tid] = 0;

		__syncthreads();
		if(tid * INTERPOLATION_BITS_WIDTH < num_bits) {
			int i = tid * INTERPOLATION_BITS_WIDTH;
			multiply_unrolled<INTERPOLATION_TOWER_HEIGHT>(xor_of_halves + i, coefficient, folded_points + i);
		}
		
		__syncthreads();
		folded_points[tid] = folded_points[tid] ^ lower_batches[idx];
		__syncthreads();
		if(j > 0) {
			multiply_thread(composition, folded_points, composition, num_bits, tid, 0);
		} else {
			composition[tid] = folded_points[tid];
		}
		__syncthreads();
	}
	
	atomicXor(destination + tid, composition[tid]);
}

template <uint32_t INTERPOLATION_POINTS, uint32_t COMPOSITION_SIZE, uint32_t EVALS_PER_MULTILINEAR>
void compute_compositions_fine( // evaluates Si(Xi) at multiple points and gets the claimed sum
	const uint32_t* multilinear_evaluations, // d x 2^n table representing the 3 multiplied hypercubes
	uint32_t* multilinear_products_sums, 
	uint32_t* folded_products_sums,
	const uint32_t coefficients[INTERPOLATION_POINTS * BITS_WIDTH],
	const uint32_t num_batch_rows,
	const cudaStream_t streams[INTERPOLATION_POINTS + 1]
) {

	cudaMemset(multilinear_products_sums, 0, BITS_WIDTH * sizeof(uint32_t));
	cudaMemset(folded_products_sums, 0, INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t));

	//printf("launch %d blocks of %d threads", num_batch_rows, BITS_WIDTH);
	composition_then_add_kernel<<<num_batch_rows, BITS_WIDTH>>>(multilinear_evaluations, multilinear_products_sums, BITS_WIDTH, COMPOSITION_SIZE);	

	for(int i = 0; i < INTERPOLATION_POINTS; i++) {
		const uint32_t* coefficient = coefficients + BITS_WIDTH * i;
		uint32_t* destination = folded_products_sums + BITS_WIDTH * i;

		interpolation_then_composition_then_add
			<<<num_batch_rows / 2, BITS_WIDTH>>>(multilinear_evaluations, coefficient, destination, BITS_WIDTH, num_batch_rows, COMPOSITION_SIZE);	
	}

	check(cudaDeviceSynchronize());
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

	for (uint32_t row_idx = tid; row_idx < num_batch_rows; row_idx += gridDim.x * blockDim.x) {
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