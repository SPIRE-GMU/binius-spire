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
}

void test_calculate_random_challenge_products() {
    const uint32_t round_idx = 2;
    const uint32_t d = 2;
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
        
        printf("expected = %x %x %x %x ", expected[0], expected[1], expected[2], expected[3]);
        printf("res = %x %x %x %x\n", res[i*4], res[i*4+1], res[i*4+2], res[i*4+3]);
    }
}

TEST_CASE("test_calculate_random_challenge_products") {
    for(int i = 0; i < 1; i++) {
        test_calculate_random_challenge_products();
    }
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
    res1 = tower_height_7_mul(a, b);
    mult128_unbitsliced(a_arr, b_arr, res2); // tested, works fine

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