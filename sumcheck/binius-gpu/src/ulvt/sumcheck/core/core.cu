#include <cstdint>
#include <iostream>

#include "../../finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "../../finite_fields/circuit_generator/unrolled/binary_tower_rolled.cuh"
#include "../../utils/bitslicing.cuh"
#include "../utils/constants.hpp"
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

#define BLOCK_SIZE_MULTILINEAR 1024
#define MAX_SMEM_SIZE 32
#define COARSE_FACTOR BLOCK_SIZE_MULTILINEAR / MAX_SMEM_SIZE
#define LINE printf("%s: at line %d\n", __FILE__, __LINE__)


__host__ __device__ void add_ee(const uint32_t a[BITS_WIDTH], const uint32_t b[BITS_WIDTH], uint32_t destination[BITS_WIDTH]){
	for(int i = 0; i < BITS_WIDTH; i++) {
		destination[i] = a[i] ^ b[i];
	}
}

__host__ __device__ void calculate_es(
	const uint32_t e[BITS_WIDTH], 
	const uint32_t s[INTERPOLATION_BITS_WIDTH],
	uint32_t destination[BITS_WIDTH]
) { // product of extension field batch and interpolation field batch
	for(int i = 0; i < BITS_WIDTH; i += INTERPOLATION_BITS_WIDTH) {
		multiply_unrolled<INTERPOLATION_TOWER_HEIGHT>(e + i, s, destination + i);
	}
}

__host__ __device__ void calculate_eb(
	const uint32_t e[BITS_WIDTH], 
	const uint32_t b,
	uint32_t destination[BITS_WIDTH]
) { // product of extension field batch and interpolation field batch
	for(int i = 0; i < BITS_WIDTH; i++) {
		destination[i] = e[i] & b;
	}
}

__host__ __device__ void calculate_eb_xor(
	const uint32_t e[BITS_WIDTH], 
	const uint32_t b,
	uint32_t destination[BITS_WIDTH]
) { // product of extension field batch and interpolation field batch
	for(int i = 0; i < BITS_WIDTH; i++) {
		destination[i] ^= e[i] & b;
	}
}

__device__ void calculate_eb_atomic(
	const uint32_t e[BITS_WIDTH], 
	const uint32_t b,
	uint32_t destination[BITS_WIDTH]
) { // product of extension field batch and interpolation field batch
	for(int i = 0; i < BITS_WIDTH; i++) {
		atomicXor(destination + i, e[i] & b);
	}
}

__global__ void calculate_multilinear_product_sums_kernel( // can possibly tile becuase a lot of data reuse
	const uint32_t* multilinear_evaluations_p1_unbitsliced,
	const uint32_t* multilinear_evaluations,
	uint32_t* destination,
	const uint32_t d,
	const uint32_t round_idx,
	const uint32_t n
) { // each thread computes 1 batch
	//__shared__ uint32_t products_s[BLOCK_SIZE_MULTILINEAR][128];
	__shared__ uint32_t products_s[MAX_SMEM_SIZE * BITS_WIDTH];

	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	uint32_t num_terms = (1U << (d*round_idx + d));  // n=28, d=4, round_idx=4
	uint32_t num_batches_in_product = (1U << (n - round_idx - 1)) / 32;
	uint32_t start_pos_in_table[MAX_COMPOSITION_SIZE];
	//uint32_t eb_product[BITS_WIDTH];
	uint32_t eb_sum[INTS_PER_VALUE];

	for(uint64_t k = idx; k < (uint64_t) num_terms * num_batches_in_product; k += stride){
		uint32_t i = k / num_batches_in_product;
		uint32_t x_idx = k % num_batches_in_product;

		uint32_t product_sum_batch = 0;
		memset(start_pos_in_table, 0, d*sizeof(uint32_t));
		for(int j = 0; j < d*round_idx + d; j++) {
			int p_idx = j / (round_idx + 1);
			int var_idx = j % (round_idx + 1);
			start_pos_in_table[p_idx] += ((i & (1 << j)) != 0) << (n - 1 - var_idx);
		}
		uint32_t product = 0xFFFFFFFF;
		for(int p_idx = 0; p_idx < d-1; p_idx++) {
			product = product & multilinear_evaluations[(p_idx) * (1 << n) / 32 + start_pos_in_table[p_idx] / 32 + x_idx];
		}
		
		for(int l = threadIdx.x; l < MAX_SMEM_SIZE * BITS_WIDTH; l += blockDim.x) {
			products_s[l] = 0;
		}

		__syncthreads();
		
		memset(eb_sum, 0, 4 * sizeof(uint32_t));
		for(int j = 0; j < 32; j++) {
			if(product & (1 << j)) {
				for(int l = 0; l < 4; l++) {
					eb_sum[l] ^= multilinear_evaluations_p1_unbitsliced[x_idx * BITS_WIDTH + start_pos_in_table[d - 1] * INTS_PER_VALUE + j * INTS_PER_VALUE + l];
				}
			}
		}

		for(int j = 0; j < BITS_WIDTH; j++) {
			int limb_idx = j / 32;
			int bit_idx = j % 32;
			atomicOr(products_s + threadIdx.x / 32 * BITS_WIDTH + j, ((eb_sum[limb_idx] >> bit_idx) & 1) << (threadIdx.x % 32));
		}

		__syncthreads();

		for(int reduce_stride = MAX_SMEM_SIZE / 2; reduce_stride > 0; reduce_stride /= 2) {
			if(threadIdx.x < reduce_stride) {
				add_ee(products_s + threadIdx.x * BITS_WIDTH, products_s + (threadIdx.x + reduce_stride) * BITS_WIDTH, products_s + threadIdx.x * BITS_WIDTH);
			}
			__syncthreads();
		}

		if(threadIdx.x < BITS_WIDTH) {
			uint32_t destination_batch_idx = i / 32 * BITS_WIDTH;
			uint32_t destination_batch_bit_idx = i % 32;
			atomicXor(destination + destination_batch_idx + threadIdx.x, (__builtin_popcount(products_s[threadIdx.x]) & 1) << destination_batch_bit_idx);
		}
		__syncthreads();
	}
}


__host__ __device__ void calculate_random_challenge_products(
	const uint32_t* random_challenges,
	uint32_t* destination,
	const uint32_t d,
	const uint32_t round_idx
) {
	uint32_t num_terms = (1 << (d*round_idx + d));
	for(int i = 0; i < num_terms; i += 32) { // batches of 32 bit ints
		uint32_t batch_product[BITS_WIDTH];
		batch_product[0] = 0xFFFFFFFF;
		for(int j = 1; j < BITS_WIDTH; j++) {
			batch_product[j] = 0;
		}
		//printf("i %d\n", i);
		for(int r_idx = 0; r_idx < d*round_idx + d; r_idx++) {
			if(r_idx % (round_idx + 1) == round_idx) continue; // skip if it's a c index (* interpolation point, not random challenge)
			uint32_t random_challenge_idx = r_idx % (round_idx + 1);
			uint32_t batch[BITS_WIDTH];
			memset(batch, 0, BITS_WIDTH*sizeof(uint32_t));

			for(int term_idx = i; term_idx < i+32; term_idx++) {
				uint32_t first_bit_xor = (term_idx & (1 << r_idx)) == 0;
				batch[0] ^= ((random_challenges[random_challenge_idx * INTS_PER_VALUE] & 1) ^ first_bit_xor) << (term_idx - i);
				for(int bit_idx = 1; bit_idx < BITS_WIDTH; bit_idx++) {
					int limb_idx = bit_idx / 32;
					int bit_in_limb_idx = bit_idx % 32;
					batch[bit_idx] ^= ((random_challenges[random_challenge_idx * INTS_PER_VALUE + limb_idx] >> bit_in_limb_idx) & 1) << (term_idx - i);
				}
			}

			multiply_unrolled<TOWER_HEIGHT>(batch_product, batch, batch_product);
		}

		memcpy(destination + BITS_WIDTH * i / 32, batch_product, BITS_WIDTH * sizeof(uint32_t));
	}
}

__host__ __device__ void calculate_interpolation_point_products(
	const uint32_t interpolation_point,
	uint32_t* destination,
	const uint32_t d,
	const uint32_t round_idx	
) {
	uint32_t num_terms = (1 << (d*round_idx + d));
	//printf("num_terms %d\n", num_terms);
	for(int i = 0; i < num_terms; i += 32) {
		uint32_t batch_product[INTERPOLATION_BITS_WIDTH];
		batch_product[0] = 0xFFFFFFFF;
		for(int j = 1; j < INTERPOLATION_BITS_WIDTH; j++) {
			batch_product[j] = 0;
		}
		for(int r_idx = 0; r_idx < d*round_idx + d; r_idx++) {
			if(r_idx % (round_idx + 1) != round_idx) continue; // only do interpolation points
			uint32_t batch[INTERPOLATION_BITS_WIDTH];
			memset(batch, 0, INTERPOLATION_BITS_WIDTH * sizeof(uint32_t));

			for(int term_idx = i; term_idx < min(i+32, num_terms); term_idx++) {
				uint32_t first_bit_xor = (term_idx & (1 << r_idx)) == 0;
				batch[0] ^= ((interpolation_point & 1) ^ first_bit_xor) << (term_idx - i);
				//printf("term_idx %d\n", term_idx);
				//printf("%d", (batch[0] >> (term_idx-i)) & 1);
				for(int bit_idx = 1; bit_idx < INTERPOLATION_BITS_WIDTH; bit_idx++) {
					//printf("%d", (batch[bit_idx] >> (term_idx-i)) & 1);
					batch[bit_idx] ^= ((interpolation_point >> bit_idx) & 1) << (term_idx - i);
				}
				//printf("\n");
			}

			//printf("(%d %d %d %d) * (%d %d %d %d)\n", batch_product[0], batch_product[1], batch_product[2], batch_product[3], batch[0], batch[1], batch[2], batch[3]);
			multiply_unrolled<INTERPOLATION_TOWER_HEIGHT>(batch_product, batch, batch_product);

			//printf("multiply result %d %d %d %d\n", batch_product[0], batch_product[1], batch_product[2], batch_product[3]);
		}

		//printf("memcpy destination + %d out of %d ints\n", INTERPOLATION_BITS_WIDTH*i / 32, (num_terms + 31) / 32 * INTERPOLATION_BITS_WIDTH);
		memcpy(destination + INTERPOLATION_BITS_WIDTH*i / 32, batch_product, INTERPOLATION_BITS_WIDTH*sizeof(uint32_t));
	}
}

__host__ __device__ void calculate_multilinear_product_sums(
	const uint32_t* multilinear_evaluations_p1, // eq polynomial in zerocheck (F(2^128))
	const uint32_t* multilinear_evaluations,
	uint32_t* destination,
	const uint32_t d,
	const uint32_t round_idx,
	const uint32_t n
) {
	uint32_t num_terms = (1 << (d*round_idx + d));
	uint32_t num_batches_in_product = (1 << (n - round_idx - 1)) / 32;
	for(int i = 0; i < num_terms; i++) {
		uint32_t product_sum_batch = 0;
		//uint32_t start_points[d][round_idx+1];
		//uint32_t start_pos_in_table[d];
		uint32_t* start_pos_in_table = (uint32_t*) malloc(d * sizeof(uint32_t));
		memset(start_pos_in_table, 0, d*sizeof(uint32_t));
		for(int j = 0; j < d*round_idx + d; j++) {
			int p_idx = j / (round_idx + 1);
			int x_idx = j % (round_idx + 1);
			start_pos_in_table[p_idx] += ((i & (1 << j)) != 0) << (n - 1 - x_idx);
		}
		for(int x_idx = 0; x_idx < num_batches_in_product; x_idx++) {
			uint32_t product = 0xFFFFFFFF;
			for(int p_idx = 0; p_idx < d; p_idx++) {
				product = product & multilinear_evaluations[p_idx * (1 << n) / 32 + start_pos_in_table[p_idx] / 32 + x_idx];
			}
			product_sum_batch ^= product;
		}
		
		uint32_t destination_batch_idx = i / 32;
		uint32_t destination_batch_bit_idx = i % 32;
		destination[destination_batch_idx] ^= (__builtin_popcount(product_sum_batch) % 2) << destination_batch_bit_idx; // XOR of all bits in a batch sum

	}
}

__host__ void calculate_interpolation_points(
	const uint32_t* multilinear_evaluations_p1_unbitsliced,
	const uint32_t* multilinear_evaluations_p1, // eq polynomial in zerocheck (F(2^128))
	const uint32_t* multilinear_evaluations_d, // should be on gpu
	const uint32_t* random_challenges,
	uint32_t* destination,
	uint32_t* claimed_sum,
	const uint32_t d,
	const uint32_t round_idx,
	const uint32_t n
) {
	uint32_t num_terms = (1 << (d*round_idx + d));
	uint32_t* multilinear_product_sums_d;
	uint32_t* multilinear_product_sums = (uint32_t*) malloc((num_terms + 31) / 32 * BITS_WIDTH * sizeof(uint32_t));// [num_terms / 32];
	uint32_t* random_challenge_products = (uint32_t*) malloc((num_terms + 31) / 32 * BITS_WIDTH * sizeof(uint32_t)); //[num_terms / 32 * BITS_WIDTH];
	uint32_t* terms = (uint32_t*) malloc((num_terms + 31) / 32 * BITS_WIDTH * sizeof(uint32_t));
	uint32_t* interpolation_point_products = (uint32_t*) malloc((num_terms + 31) / 32 * INTERPOLATION_BITS_WIDTH * sizeof(uint32_t));

	check(cudaMalloc(&multilinear_product_sums_d, (num_terms + 31) / 32 * BITS_WIDTH * sizeof(uint32_t)));
	memset(terms, 0, (num_terms + 31) / 32 * BITS_WIDTH * sizeof(uint32_t));	
	memset(multilinear_product_sums, 0, (num_terms + 31) / 32 * BITS_WIDTH * sizeof(uint32_t));
	memset(claimed_sum, 0, INTS_PER_VALUE * sizeof(uint32_t));

	cudaMemcpy(multilinear_product_sums_d, multilinear_product_sums, (num_terms + 31) / 32 * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice);

	uint32_t batches_per_term = (1 << (n - round_idx - 1)) / 32;
	uint32_t num_blocks = max(batches_per_term / BLOCK_SIZE_MULTILINEAR, 8192);
	calculate_multilinear_product_sums_kernel<<<num_blocks, BLOCK_SIZE_MULTILINEAR>>>(multilinear_evaluations_p1_unbitsliced, multilinear_evaluations_d, multilinear_product_sums_d, d, round_idx, n);
	check(cudaDeviceSynchronize());
	calculate_random_challenge_products(random_challenges, random_challenge_products, d, round_idx);	

	cudaMemcpy(multilinear_product_sums, multilinear_product_sums_d, (num_terms + 31) / 32 * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	for(int interpolation_point = 0; interpolation_point < d+1; interpolation_point++) {
		calculate_interpolation_point_products(interpolation_point, interpolation_point_products, d, round_idx);

		for(int i = 0; i < (num_terms+31) / 32; i++) {
			calculate_es(random_challenge_products + i*BITS_WIDTH, interpolation_point_products + i*INTERPOLATION_BITS_WIDTH, terms + i*BITS_WIDTH);
			multiply_unrolled<TOWER_HEIGHT>(terms + i*BITS_WIDTH, multilinear_product_sums + i*BITS_WIDTH, terms + i*BITS_WIDTH);
		}
		
		uint32_t batch_sum[BITS_WIDTH];
		uint32_t res[INTS_PER_VALUE];
		memset(batch_sum, 0, BITS_WIDTH * sizeof(uint32_t));
		memset(res, 0, INTS_PER_VALUE * sizeof(uint32_t));
		for(int term_idx = 0; term_idx < (num_terms + 31) / 32 * BITS_WIDTH; term_idx++) {
			batch_sum[term_idx % BITS_WIDTH] ^= terms[term_idx];
		}
		for(int bit_idx = 0; bit_idx < BITS_WIDTH; bit_idx++) {
			int limb_idx = bit_idx / 32;
			int bit_in_limb_idx = bit_idx % 32;
			res[limb_idx] ^= (__builtin_popcount(batch_sum[bit_idx]) % 2) << bit_in_limb_idx;
		}
		memcpy(destination + interpolation_point*INTS_PER_VALUE, res, INTS_PER_VALUE * sizeof(uint32_t));
	}

	for(int i = 0; i < INTS_PER_VALUE; i++) {
		// claimed = S(0) + S(1)
		claimed_sum[i] = destination[i] ^ destination[i + INTS_PER_VALUE];
	}
}

__host__ __device__ void get_batch(
	const uint32_t* multilinear_evaluations_d,
	const uint32_t* random_challenges_subset_products, 
	uint32_t* destination,
	const uint32_t idx,
	const uint32_t p_idx,
	const uint32_t round_idx,
	const uint32_t n
) {
	uint32_t stride = (1 << (n - round_idx)) / 32;
	uint32_t j = 0;
	memset(destination, 0, BITS_WIDTH * sizeof(uint32_t));
	for(int i = idx; i < (1 << n) / 32; i += stride) {
		printf("i=%d out of %d, batch=%d\n", i, (1 << n) / 32, multilinear_evaluations_d[i + p_idx * (1 << n) / 32]);
		calculate_eb_xor(random_challenges_subset_products + j * BITS_WIDTH, multilinear_evaluations_d[i + p_idx * (1 << n) / 32], destination);
		j++;
	}
}

__global__ void fold_multiple_kernel( // only binary
	const uint32_t* multilinear_evaluations_d,
	uint32_t* destination,
	uint32_t* random_challenge_subset_products,
	const uint32_t round_idx,
	const uint32_t n,
	const uint32_t d
) {
	uint32_t num_batches = (1 << n) / 32;
	uint32_t out_num_batches = (1 << (n - round_idx)) / 32;
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	for(int i = idx; i < d * out_num_batches; i += blockDim.x*gridDim.x) {
		uint32_t p_idx = i / out_num_batches;
		uint32_t batch_idx = i % out_num_batches;
		get_batch(multilinear_evaluations_d, random_challenge_subset_products, destination + i, batch_idx, p_idx, round_idx, n);
		/*if(i == 0) {
			printf("folded batch\n");
			for(int j = 0; j < BITS_WIDTH; j++) 
				printf("%x\n", destination[j]);
		}*/
	}
}

__host__ void fold_multiple(
	const uint32_t* folded_multilinear_evaluations_p1,
	const uint32_t* multilinear_evaluations_d,
	const uint32_t* random_challenges,
	uint32_t* destination,
	const uint32_t round_idx,
	const uint32_t n,
	const uint32_t d
) {
	uint32_t* random_challenges_copy = (uint32_t*) malloc((1 << round_idx) * BITS_WIDTH * sizeof(uint32_t));
	memcpy(random_challenges_copy, random_challenges, (1 << round_idx) * BITS_WIDTH * sizeof(uint32_t));
	//uint32_t random_challenge_subset_products[BITS_WIDTH * (1 << ROUND_IDX)];
	uint32_t batches_per_multilinear = (1 << (n - round_idx)) / 32;
	uint32_t* random_challenge_subset_products = (uint32_t*) malloc(BITS_WIDTH * (1 << round_idx) * sizeof(uint32_t));
	for(int mask = 0; mask < (1 << round_idx); mask++) {
		uint32_t* product = random_challenge_subset_products + mask*BITS_WIDTH;
		for(int i = 0; i < BITS_WIDTH; i++) {
			if(i == 0)
				product[i] = 0xFFFFFFFF;
			else
				product[i] = 0;
		}
		for(int i = 0; i < round_idx; i++) {
			if(((mask >> i) & 1) == 0) {
				random_challenges_copy[BITS_WIDTH*i] ^= 0xFFFFFFFF;
				multiply_unrolled<TOWER_HEIGHT>(product, random_challenges_copy + BITS_WIDTH * i, product);
				random_challenges_copy[BITS_WIDTH*i] ^= 0xFFFFFFFF;
			} else {
				multiply_unrolled<TOWER_HEIGHT>(product, random_challenges_copy + BITS_WIDTH * i, product);
			}
		}

		// for(int i = 0; i < BITS_WIDTH; i++) {
		// 	printf("%x", product[i] & 1);
		// }
		// printf("\n");
	}


	uint32_t* random_challenge_subset_products_d;
	check(cudaMalloc(&random_challenge_subset_products_d, BITS_WIDTH * (1 << round_idx) * sizeof(uint32_t)));
	check(cudaMemcpy(random_challenge_subset_products_d, random_challenge_subset_products, BITS_WIDTH * (1 << round_idx) * sizeof(uint32_t), cudaMemcpyHostToDevice));
	
	check(cudaMemcpy(destination, folded_multilinear_evaluations_p1, batches_per_multilinear * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
	fold_multiple_kernel<<<8192, 256>>>(multilinear_evaluations_d, destination + batches_per_multilinear * BITS_WIDTH, random_challenge_subset_products_d, round_idx, n, d-1);

	check(cudaDeviceSynchronize());
}

__host__ __device__ void evaluate_composition_on_batch_row( // after folding, calculate the claimed sum over hypercube by multiplying the individual multilinear evaluations
	const uint32_t* first_batch_of_row,
	uint32_t* batch_composition_destination,
	const uint32_t composition_size,
	const uint32_t original_evals_per_col
) {
	memcpy(batch_composition_destination, first_batch_of_row, BITS_WIDTH * sizeof(uint32_t));

	for (int operand_in_composition = 1; operand_in_composition < composition_size; ++operand_in_composition) {
		// next polynomial in the composition
		// makes sense because the INTS_PER_VALUE still represents the size of all the batches
		const uint32_t* nth_batch_of_row =
			first_batch_of_row + operand_in_composition * original_evals_per_col * INTS_PER_VALUE; // move forward to the next polynomial in the composition
		
		// multiply
		multiply_unrolled<TOWER_HEIGHT>(batch_composition_destination, nth_batch_of_row, batch_composition_destination);
	}
}

__host__ __device__ void fold_batch( // fold polynomial table in half by plugging in random challenge point
	const uint32_t lower_batch[BITS_WIDTH],
	const uint32_t upper_batch[BITS_WIDTH],
	uint32_t dst_batch[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH], // coef is actually just 1 value (r_i) copied over; this makes it so that bitslicing works natively with multiplciations here
	const bool is_interpolation
) {
	uint32_t xor_of_halves[BITS_WIDTH];

	for (int i = 0; i < BITS_WIDTH; ++i) {
		xor_of_halves[i] = lower_batch[i] ^ upper_batch[i];
	}

	uint32_t product[BITS_WIDTH];
	memset(product, 0, BITS_WIDTH * sizeof(uint32_t));

	// Multiply chunk-wise based on field height of coefficient
	// For random challenges this will be the full 7
	// For interpolation points this will be no more than 2

	if (is_interpolation) {
		for (int i = 0; i < BITS_WIDTH; i += INTERPOLATION_BITS_WIDTH) {
			multiply_unrolled<INTERPOLATION_TOWER_HEIGHT>(xor_of_halves + i, coefficient, product + i);
		}
	} else {
		multiply_unrolled<TOWER_HEIGHT>(xor_of_halves, coefficient, product);
	}

	for (int i = 0; i < BITS_WIDTH; ++i) {
		dst_batch[i] = lower_batch[i] ^ product[i];
	}
}

__host__ __device__ void fold_batch_1( // fold polynomial table in half by plugging in random challenge point
	const uint32_t lower_batch[BITS_WIDTH],
	const uint32_t upper_batch[BITS_WIDTH],
	uint32_t dst_batch[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH], // coef is actually just 1 value (r_i) copied over; this makes it so that bitslicing works natively with multiplciations here
	const bool is_interpolation
) {
	uint32_t tmp_coef[BITS_WIDTH];
	uint32_t upper_product[BITS_WIDTH];
	uint32_t lower_product[BITS_WIDTH];
	memcpy(tmp_coef, coefficient, BITS_WIDTH * sizeof(uint32_t));

	multiply_unrolled<TOWER_HEIGHT>(upper_batch, tmp_coef, upper_product);
	tmp_coef[0] ^= 0xFFFFFFFF;
	multiply_unrolled<TOWER_HEIGHT>(lower_batch, tmp_coef, lower_product);

	for(int i = 0; i < BITS_WIDTH; i++) {
		dst_batch[i] = upper_product[i] ^ lower_product[i];
	}
}


void fold_small(
	const uint32_t source[BITS_WIDTH],
	uint32_t destination[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const uint32_t list_len
) {
	uint32_t half_len = list_len / 2;

	uint32_t batch_to_be_multiplied[BITS_WIDTH];

	memcpy(batch_to_be_multiplied, source, BITS_WIDTH * sizeof(uint32_t));

	for (int i = 0; i < BITS_WIDTH; ++i) {
		batch_to_be_multiplied[i] >>= half_len;  // Move the upper half into the lower half of this operand
		batch_to_be_multiplied[i] ^= source[i];  // Add two halves before multiplying
	}

	uint32_t product[BITS_WIDTH];

	multiply_unrolled<TOWER_HEIGHT>(batch_to_be_multiplied, coefficient, product);

	for (int i = 0; i < BITS_WIDTH; ++i) {
		destination[i] = source[i] ^ product[i];
	}
}

__host__ __device__ void compute_sum( // sum the compositions to obtain Si(Xi) or the claimed sum
	uint32_t sum[INTS_PER_VALUE],
	uint32_t bitsliced_batch[BITS_WIDTH],
	const uint32_t num_eval_points_being_summed_unpadded
) {
	BitsliceUtils<BITS_WIDTH>::bitslice_untranspose(bitsliced_batch);

	memset(sum, 0, INTS_PER_VALUE * sizeof(uint32_t));

	for (uint32_t i = 0; i < min(BITS_WIDTH, INTS_PER_VALUE * num_eval_points_being_summed_unpadded); ++i) {
		sum[i % INTS_PER_VALUE] ^= bitsliced_batch[i];
	}
}