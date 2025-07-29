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
#include "../../utils/bitslicing.cuh"

#define PRINTLINE printf("%d\n", __LINE__);

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

void test_composition_then_add_kernel() {
    uint32_t x[128*4];// = malloc(128 * 4 * sizeof(uint32_t));
    uint32_t solution[128];// = malloc(128 * sizeof(uint32_t)); 
    uint32_t output[128];// = malloc(128 * sizeof(uint32_t)); 
    uint32_t composition[128*2];// = malloc(128 * 2 * sizeof(uint32_t));

    uint32_t* x_d;
    uint32_t* output_d;

    cudaMalloc((void**) &x_d, 128 * 4 * sizeof(uint32_t));
    cudaMalloc((void**) &output_d, 128*sizeof(uint32_t));

    for(int i = 0; i < 128 * 4; i++) {
        if(i < 128) {
            solution[i] = 0;
            output[i] = 0;
        }
        if(i < 2*128) {
            composition[i] = 0xFFFFFFFF;
        }
        x[i] = std::rand();
    }

    cudaMemcpy(x_d, x, 128*4*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(output_d, output, 128*sizeof(uint32_t), cudaMemcpyHostToDevice);

    for(int j = 0; j < 2; j++) {
        //printf("%d %d\n", x[j*128], x[j*128+256]);
        evaluate_composition_on_batch_row(x + j * 128, composition + j * 128, 2, 64);
    }

    for(int j = 0; j < 2; j++) {
        for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
            solution[i] ^= composition[i + j*128]; // add to running batch sums
        }
    }

    composition_then_add_kernel<<<2, 128>>>(x_d, output_d, 128, 2);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, 128*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 128; i++) {
        REQUIRE(output[i] == solution[i]);
    }
}

TEST_CASE("composition_then_add_kernel") {
    test_composition_then_add_kernel();
    test_composition_then_add_kernel();
    test_composition_then_add_kernel();
    test_composition_then_add_kernel();
    test_composition_then_add_kernel();
}

void test_interpolation_then_composition_then_add() {
    uint32_t x[128*4];// = malloc(128 * 4 * sizeof(uint32_t));
    uint32_t coefficient[128];
    uint32_t solution[128];// = malloc(128 * sizeof(uint32_t)); 
    uint32_t output[128];// = malloc(128 * sizeof(uint32_t)); 
    uint32_t composition[128];// = malloc(128 * 2 * sizeof(uint32_t));
    uint32_t folded[128*2];// = malloc(128 * 2 * sizeof(uint32_t));

    uint32_t* x_d;
    uint32_t* coefficient_d;
    uint32_t* output_d;

    check(cudaMalloc((void**) &x_d, 128 * 4 * sizeof(uint32_t)));
    check(cudaMalloc((void**) &output_d, 128*sizeof(uint32_t)));
    check(cudaMalloc((void**) &coefficient_d, 128*sizeof(uint32_t)));

    for(int i = 0; i < 128 * 4; i++) {
        if(i < 128) {
            solution[i] = 0;
            output[i] = 0;
            coefficient[i] = 0;
            composition[i] = 0xFFFFFFFF;
        }
        x[i] = std::rand();
    }

    check(cudaMemcpy(x_d, x, 128*4*sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(output_d, output, 128*sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(coefficient_d, coefficient, 128*sizeof(uint32_t), cudaMemcpyHostToDevice));

    for(int j = 0; j < 2; j++) {
        fold_batch(x + j*256, x + 128 + j*256, folded + j*128, coefficient, true); 
    }
    evaluate_composition_on_batch_row(folded, composition, 2, 32);
    for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
        solution[i] ^= composition[i]; // add to running batch sums
    }

    interpolation_then_composition_then_add<<<1, 128>>>(x_d, coefficient_d, output_d, 128, 2, 2);
    check(cudaDeviceSynchronize());

    check(cudaMemcpy(output, output_d, 128*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for(int i = 0; i < 128; i++) {
        REQUIRE(output[i] == solution[i]);
    }

}

TEST_CASE("interpolation_then_composition_then_add_kernel") {
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
}
/*
TEST_CASE("test_calculate_multilinear_product_sums") {
    const int batches = 4;
    const int output_sz = 1;
    std::srand(std::chrono::system_clock::now().time_since_epoch().count());
    uint32_t multilinear_evaluations[2*batches];
    for(int i = 0; i < 2*batches; i++) {
        multilinear_evaluations[i] = std::rand();
        //multilinear_evaluations[i] = 0xFFFFFFFF & 0; // Use a constant value for testing
    }
    uint32_t destination[output_sz];
    memset(destination, 0, output_sz*sizeof(uint32_t));

    calculate_multilinear_product_sums(multilinear_evaluations, destination, 2, 0, 7);

    uint32_t bit0 = destination[0] & 1;
    uint32_t bit1 = (destination[0] >> 1) & 1;
    uint32_t bit2 = (destination[0] >> 2) & 1;
    uint32_t bit3 = (destination[0] >> 3) & 1;

    REQUIRE(bit0 == __builtin_popcount((multilinear_evaluations[0] & multilinear_evaluations[4]) ^ (multilinear_evaluations[1] & multilinear_evaluations[5])) % 2);
    REQUIRE(bit1 == __builtin_popcount((multilinear_evaluations[2] & multilinear_evaluations[4]) ^ (multilinear_evaluations[3] & multilinear_evaluations[5])) % 2);
    REQUIRE(bit2 == __builtin_popcount((multilinear_evaluations[0] & multilinear_evaluations[6]) ^ (multilinear_evaluations[1] & multilinear_evaluations[7])) % 2);
    REQUIRE(bit3 == __builtin_popcount((multilinear_evaluations[2] & multilinear_evaluations[6]) ^ (multilinear_evaluations[3] & multilinear_evaluations[7])) % 2);

    //REQUIRE(bit1 == __builtin_popcount(multilinear_evaluations[0] & (multilinear_evaluations[1] >> 1)) % 2);
    //REQUIRE
}

//TEST_CASE("test_calculate_multilinear_product_sums_round1") {
void test_calculate_multilinear_product_sums_round1() {
    const int batches = 4;
    const int output_sz = 1;
    std::srand(std::chrono::system_clock::now().time_since_epoch().count());
    uint32_t multilinear_evaluations[2*batches];
    for(int i = 0; i < 2*batches; i++) {
        multilinear_evaluations[i] = std::rand();
        //multilinear_evaluations[i] = 0xFFFFFFFF & 0; // Use a constant value for testing
    }
    uint32_t destination[output_sz];
    memset(destination, 0, output_sz*sizeof(uint32_t));

    calculate_multilinear_product_sums(multilinear_evaluations, destination, 2, 1, 7);

    uint32_t bit0 = destination[0] & 1;
    uint32_t bit1 = (destination[0] >> 1) & 1;
    uint32_t bit2 = (destination[0] >> 2) & 1;
    uint32_t bit3 = (destination[0] >> 3) & 1;
    uint32_t bit4 = (destination[0] >> 4) & 1;
    uint32_t bit5 = (destination[0] >> 5) & 1;
    uint32_t bit6 = (destination[0] >> 6) & 1;
    uint32_t bit7 = (destination[0] >> 7) & 1;

    REQUIRE(bit0 == __builtin_popcount((multilinear_evaluations[0] & multilinear_evaluations[4])) % 2);
    REQUIRE(bit1 == __builtin_popcount((multilinear_evaluations[2] & multilinear_evaluations[4])) % 2);
    REQUIRE(bit2 == __builtin_popcount((multilinear_evaluations[1] & multilinear_evaluations[4])) % 2);
    REQUIRE(bit3 == __builtin_popcount((multilinear_evaluations[3] & multilinear_evaluations[4])) % 2);
    REQUIRE(bit4 == __builtin_popcount((multilinear_evaluations[0] & multilinear_evaluations[6])) % 2);
    REQUIRE(bit5 == __builtin_popcount((multilinear_evaluations[2] & multilinear_evaluations[6])) % 2);
    REQUIRE(bit6 == __builtin_popcount((multilinear_evaluations[1] & multilinear_evaluations[6])) % 2);
    REQUIRE(bit7 == __builtin_popcount((multilinear_evaluations[3] & multilinear_evaluations[6])) % 2);
}

TEST_CASE("test_calculate_multilinear_product_sums_round1") {
    for(int i = 0; i < 100; i++)
        test_calculate_multilinear_product_sums_round1();
}*/

void test_calculate_random_challenge_products() {
    const uint32_t round_idx = 4;
    const uint32_t d = 3;
    uint32_t random_challenges[round_idx * INTS_PER_VALUE];
    for(uint32_t i = 0; i < round_idx * INTS_PER_VALUE; i++) {
        random_challenges[i] = std::rand();
    }
    uint32_t num_terms = (1 << (d * round_idx + d));
    uint32_t num_batches = (num_terms + 31) / 32;
    uint32_t destination[(num_terms + 31) / 32 * BITS_WIDTH];
    uint32_t res[INTS_PER_VALUE * num_terms];
    printf("num_terms = %d, num_batches = %d\n", num_terms, num_batches);
    memset(destination, 0, sizeof(destination));
    calculate_random_challenge_products(random_challenges, destination, d, round_idx);
    untranspose128(destination, (num_terms+31)/32, res);


    for(int i = 0; i < num_terms; i++) {
        uint32_t expected[4] = {1, 0, 0, 0};
        for(uint32_t j = 0; j < d * round_idx + d; j++) {
            if(j % (round_idx+1) == round_idx) continue;
            uint32_t r_idx = j % (round_idx + 1);
            uint32_t r[4];
            memcpy(r, random_challenges + r_idx * INTS_PER_VALUE, 4 * sizeof(uint32_t));
            //printf("%d %d %d\n", i,j, i & (1 << j), i & (1 << j) == 0);
            if((i & (1 << j)) == 0) {
                //printf("286 %d %d\n", i, j);
                r[0] ^= 1;
            }
            mult128_unbitsliced(r, expected, expected);
        }
        
        //printf("expected = %x %x %x %x ", expected[0], expected[1], expected[2], expected[3]);
        //printf("res = %x %x %x %x\n", res[i*4], res[i*4+1], res[i*4+2], res[i*4+3]);
        REQUIRE(expected[0] == res[i*4]);
        REQUIRE(expected[1] == res[i*4+1]);
        REQUIRE(expected[2] == res[i*4+2]);
        REQUIRE(expected[3] == res[i*4+3]);
    }
}

TEST_CASE("test_calculate_random_challenge_products") {
    for(int i = 0; i < 10; i++) {
        test_calculate_random_challenge_products();
    }
}

void test_calculate_interpolation_point_products(uint32_t interpolation_point) {
    const uint32_t d = 2;
    const uint32_t round_idx = 2; 
    //uint32_t destination[];

    
}


TEST_CASE("test_unbitsliced_mul_int128") {
    __uint128_t a = 0;
    __uint128_t b = 0;
    uint32_t a_arr[4];
    uint32_t b_arr[4];
    uint32_t res2[4];
    __uint128_t res = 0;
    __uint128_t res1 = 0;
    uint64_t a_tmp = 0;
    for(int i = 0; i < 4; i++) {
        uint32_t a_limb = std::rand();
        uint32_t b_limb = std::rand();
        a += ((__uint128_t) a_limb) << (i*32);
        b += ((__uint128_t) b_limb) << (i*32);
        a_arr[i] = a_limb;
        b_arr[i] = b_limb;
        //a_tmp += std::rand() << (i*32);
    }
    //printf("a_tmp %llu\n", a_tmp);
    mult128_unbitsliced_no_limb(a, b, res);
    //mult128_unbitsliced_no_limb(a, res, res);
    res1 = tower_height_7_mul(a, b);
    mult128_unbitsliced(a_arr, b_arr, res2); // tested, works fine
    //mult128_unbitsliced(a_arr, res2, res2); 

    char a_str[33];
    char b_str[33];
    char res_str[33];
    char res1_str[33];
    uint128_to_hex(a, a_str);
    uint128_to_hex(b, b_str);
    uint128_to_hex(res, res_str);
    uint128_to_hex(res1, res1_str);

    printf("a = 0x%s\n", a_str);
    printf("b = 0x%s\n", b_str);
    printf("a_arr = 0x%x%x%x%x\n", a_arr[0], a_arr[1], a_arr[2], a_arr[3]);
    printf("b_arr = 0x%x%x%x%x\n", b_arr[0], b_arr[1], b_arr[2], b_arr[3]);
    printf("res = 0x%s\n", res_str);
    printf("res1 = 0x%s\n", res1_str);
    printf("res2 = 0x%x%x%x%x\n", res2[0], res2[1], res2[2], res2[3]); // works fine
}

TEST_CASE("test_unbitsliced_mul") {
    uint32_t a[4];
    uint32_t b[4];
    for(int i = 0; i < 4; i++) {
        a[i] = std::rand();
        b[i] = std::rand();
    }

    uint32_t res[4];
    mult128_unbitsliced(a, b, res);
    

    printf("a = 0x%x%x%x%x\n", a[0], a[1], a[2], a[3]);
    printf("b = 0x%x%x%x%x\n", b[0], b[1], b[2], b[3]);
    printf("res = 0x%x%x%x%x\n", res[0], res[1], res[2], res[3]);
}
/*
void test_calculate_multilinear_product_sums_kernel() {
    const uint32_t d = 3;
    const uint32_t round_idx = 3;
    const uint32_t n = 28;
    uint32_t num_terms = (1 << (d*round_idx + d));

    uint32_t* multilinear_evaluations;
    uint32_t* destination;
    uint32_t* destination_kernel;
    uint32_t* multilinear_evaluations_d;
    uint32_t* destination_d;

    multilinear_evaluations = (uint32_t*) malloc(d * (1 << n) / 32 * sizeof(uint32_t));
    destination = (uint32_t*) malloc((num_terms + 31) / 32 * sizeof(uint32_t));
    destination_kernel = (uint32_t*) malloc((num_terms + 31) / 32 * sizeof(uint32_t));
    cudaMalloc(&multilinear_evaluations_d, d * (1 << n) / 32 * sizeof(uint32_t));
    cudaMalloc(&destination_d, (num_terms + 31) / 32 * sizeof(uint32_t));
    memset(destination, 0, (num_terms + 31) / 32 * sizeof(uint32_t));
    for(int i = 0; i < d * (1 << n) / 32; i++) {
        multilinear_evaluations[i] = std::rand();
    }
    cudaMemcpy(multilinear_evaluations_d, multilinear_evaluations, d * (1 << n) / 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(destination_d, destination, (num_terms + 31) / 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    calculate_multilinear_product_sums(
        multilinear_evaluations,
        destination,
        d,
        round_idx,
        n
    );

    //calculate_multilinear_product_sums_kernel<<<1024, 512>>>(
    calculate_multilinear_product_sums_kernel_tiled<<<1024, 512>>>(
        multilinear_evaluations_d,
        destination_d,
        d,
        round_idx,
        n
    );
    check(cudaDeviceSynchronize());
    
    check(cudaMemcpy(destination_kernel, destination_d, (num_terms + 31) / 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for(int i = 0; i < (num_terms + 31) / 32; i++) {
        REQUIRE(destination_kernel[i] == destination[i]);
    }
}

TEST_CASE("test_calculate_multilinear_product_sums_kernel") {
    for(int i = 0; i < 1; i++) {
        test_calculate_multilinear_product_sums_kernel();
    }
}*/

__host__ void get_batch_full(
    const uint32_t* multilinear_evaluations,
    const uint32_t* random_challenges,
    uint32_t* destination,
    const uint32_t idx,
    const uint32_t round_idx,
    const uint32_t n,
    const uint32_t d
) {
    uint32_t* random_challenges_copy = (uint32_t*) malloc(round_idx * BITS_WIDTH * sizeof(uint32_t));
	memcpy(random_challenges_copy, random_challenges, round_idx * BITS_WIDTH * sizeof(uint32_t));
    uint32_t* random_challenge_subset_products = (uint32_t*) malloc(BITS_WIDTH * (1 << round_idx) * sizeof(uint32_t));
    memset(random_challenge_subset_products, 0, BITS_WIDTH * (1 << round_idx) * sizeof(uint32_t));

    for(int i = 0; i < round_idx; i++) {
        for(int j = 0; j < BITS_WIDTH; j++) {
            printf("%x", random_challenges[i*BITS_WIDTH + j] & 1);
            //printf("%x", random_challenges_copy[i*BITS_WIDTH + j] & 1);
        }
        printf("\n");
    }

    for(int i = 0; i < round_idx; i++) {
        for(int j = 0; j < BITS_WIDTH; j++) {
            //printf("%x", random_challenges[i*BITS_WIDTH + j] & 1);
            printf("%x", random_challenges_copy[i*BITS_WIDTH + j] & 1);
        }
        printf("\n");
    }
    printf("\n");

	for(int mask = 0; mask < (1 << round_idx); mask++) {
		uint32_t* product = random_challenge_subset_products + mask*BITS_WIDTH;
		/*for(int i = 0; i < BITS_WIDTH; i++) {
			if(i == 0)
				product[i] = 0xFFFFFFFF;
			else
				product[i] = 0;
		}*/
        //product[mask * BITS_WIDTH] = 0xFFFFFFFF;
        product[0] = 0xFFFFFFFF;


    for(int i = 0; i < (1 << round_idx); i++) {
        for(int j = 0; j < BITS_WIDTH; j++) {
            printf("%x", random_challenge_subset_products[i*BITS_WIDTH + j] & 1);
        };
        printf("\n");
    }
        printf("\n");

        for(int i = 0; i < round_idx; i++) {
            int rev_i = round_idx - i - 1;
            //int rev_i = i;
			if(((mask >> i) & 1) == 0) {
				random_challenges_copy[BITS_WIDTH*rev_i] ^= 0xFFFFFFFF;
                printf("multiply:\n");
                for(int j = 0; j < BITS_WIDTH; j++) printf("%x", product[j] & 1);
                printf("\n");
                for(int j = 0; j < BITS_WIDTH; j++) printf("%x", random_challenges_copy[BITS_WIDTH*rev_i + j] & 1);
                printf("\n");
                printf("\n");

				multiply_unrolled<TOWER_HEIGHT>(product, random_challenges_copy + BITS_WIDTH * rev_i, product);
				random_challenges_copy[BITS_WIDTH*rev_i] ^= 0xFFFFFFFF;
			} else {
                printf("multiply:\n");
                for(int j = 0; j < BITS_WIDTH; j++) printf("%x", product[j] & 1);
                printf("\n");
                for(int j = 0; j < BITS_WIDTH; j++) printf("%x", random_challenges_copy[BITS_WIDTH*rev_i + j] & 1);
                printf("\n");
                printf("\n");
				multiply_unrolled<TOWER_HEIGHT>(product, random_challenges_copy + BITS_WIDTH * rev_i, product);
			}
		}
	}

    for(int i = 0; i < (1 << round_idx); i++) {
        for(int j = 0; j < BITS_WIDTH; j++) {
            printf("%x", random_challenge_subset_products[i*BITS_WIDTH + j] & 1);
        };
        printf("\n");
    }

    // uint32_t* random_challenge_subset_products_d;
	// check(cudaMalloc(&random_challenge_subset_products_d, BITS_WIDTH * (1 << round_idx) * sizeof(uint32_t)));
	// check(cudaMemcpy(random_challenge_subset_products_d, random_challenge_subset_products, BITS_WIDTH * (1 << round_idx) * sizeof(uint32_t), cudaMemcpyHostToDevice));


    get_batch(
        multilinear_evaluations,
        random_challenge_subset_products,
        destination,
        idx,
        0,
        round_idx,
        n
    );
}

TEST_CASE("test_get_batch") {
    const uint32_t n = 20;
    const uint32_t num_batches = (1 << n) / 32;
    const uint32_t round_idx = 2; // works only for 1
    
    std::srand(0);

    uint32_t* multilinear_evaluations_base = (uint32_t*) malloc(num_batches * sizeof(uint32_t));
    uint32_t* multilinear_evaluations_extension = (uint32_t*) malloc(num_batches * BITS_WIDTH * sizeof(uint32_t));
    uint32_t* random_challenge = (uint32_t*) malloc(round_idx * INTS_PER_VALUE * sizeof(uint32_t));
    uint32_t* random_challenge_batch = (uint32_t*) malloc(round_idx * BITS_WIDTH * sizeof(uint32_t));

    uint32_t* multilinear_evaluations_base_d;
    uint32_t* multilinear_evaluations_extension_d;
    uint32_t* multilinear_evaluations_folded_d;
    uint32_t* multilinear_evaluations_folded_test_d;
    uint32_t* random_challenge_d;

    check(cudaMalloc(&multilinear_evaluations_base_d, num_batches * sizeof(uint32_t)));
    check(cudaMalloc(&multilinear_evaluations_extension_d, num_batches * BITS_WIDTH * sizeof(uint32_t)));
    check(cudaMalloc(&multilinear_evaluations_folded_d, num_batches / 2 * BITS_WIDTH * sizeof(uint32_t)));
    check(cudaMalloc(&multilinear_evaluations_folded_test_d, num_batches / 2 * BITS_WIDTH * sizeof(uint32_t)));
    check(cudaMalloc(&random_challenge_d, round_idx * BITS_WIDTH * sizeof(uint32_t)));

    for(int j = 0; j < round_idx; j++) {
        for(int i = 0; i < INTS_PER_VALUE; i++) {
            random_challenge[i + j*INTS_PER_VALUE] = std::rand();
            printf("%x ",  random_challenge[i + j*INTS_PER_VALUE]);
        }
        printf("\n");
        BitsliceUtils<BITS_WIDTH>::repeat_value_bitsliced(random_challenge_batch + j*BITS_WIDTH, random_challenge + j * INTS_PER_VALUE);
    }

    check(cudaMemcpy(random_challenge_d, random_challenge_batch, round_idx * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice));

    PRINTLINE;

    for(int i = 0; i < num_batches; i++) {
        uint32_t batch = std::rand();
        multilinear_evaluations_base[i] = batch;
        multilinear_evaluations_extension[BITS_WIDTH * i] = batch;
    }
    PRINTLINE;
    check(cudaMemcpy(multilinear_evaluations_base_d, multilinear_evaluations_base, num_batches*sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(multilinear_evaluations_extension_d, multilinear_evaluations_extension, num_batches * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice));
    PRINTLINE;

    for(int i = 0; i < round_idx; i++) {
        // if(i == 0) {
        printf("fold\n");
            fold_large_list_halves<<<8192, 256>>>(
                multilinear_evaluations_extension_d,
                multilinear_evaluations_extension_d,
                random_challenge_d,
                ((1 << (n-1)) / 32) >> i,
                ((1 << n) / 32) >> i,
                ((1 << n) / 32) >> i,
                1
            );
        // } else {
        //     fold_large_list_halves<<<8192, 256>>>(
        //         multilinear_evaluations_folded_d,
        //         multilinear_evaluations_folded_d,
        //         random_challenge_d,
        //         ((1 << (n - 1)) / 32) >> i,
        //         ((1 << n) / 32) >> i,
        //         ((1 << n) / 32) >> i,
        //         1);
        // }
        check(cudaDeviceSynchronize());
    }
    PRINTLINE;

    int j = 0;
    // for(int j = 0; j < ((1 << n) / 32) >> round_idx; j++) {
        uint32_t *get_batch_destination, *fold_batch_destination;
        get_batch_destination = (uint32_t*) malloc(BITS_WIDTH * sizeof(uint32_t));
        fold_batch_destination = (uint32_t*) malloc(BITS_WIDTH * sizeof(uint32_t));

        PRINTLINE;
        get_batch_full(
            multilinear_evaluations_base,
            random_challenge_batch,
            get_batch_destination,
            j,
            round_idx,
            n,
            1 
        );

        PRINTLINE;
        check(cudaMemcpy(fold_batch_destination, multilinear_evaluations_extension_d + j*BITS_WIDTH, BITS_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for(int i = 0; i < BITS_WIDTH; i++) {
            printf("%u =?= %u\n", get_batch_destination[i], fold_batch_destination[i]);
            REQUIRE(get_batch_destination[i] == fold_batch_destination[i]);
        }
    // }

}