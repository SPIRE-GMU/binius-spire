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

__host__ void get_batch_full(
    const uint32_t* multilinear_evaluations,
    const uint32_t* random_challenges,
    uint32_t* destination,
    const uint32_t idx,
    const uint32_t round_idx,
    const uint32_t n,
    const uint32_t d
) {
    uint32_t* random_challenge_subset_products = (uint32_t*) malloc(BITS_WIDTH * (1 << round_idx) * sizeof(uint32_t));
    calculate_random_challenge_subset_products(random_challenges, random_challenge_subset_products, round_idx);

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
    const uint32_t round_idx = 3; 
    
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
        }
        BitsliceUtils<BITS_WIDTH>::repeat_value_bitsliced(random_challenge_batch + j*BITS_WIDTH, random_challenge + j * INTS_PER_VALUE);
    }

    check(cudaMemcpy(random_challenge_d, random_challenge_batch, round_idx * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice));

    for(int i = 0; i < num_batches; i++) {
        uint32_t batch = std::rand();
        multilinear_evaluations_base[i] = batch;
        multilinear_evaluations_extension[BITS_WIDTH * i] = batch;
    }
    check(cudaMemcpy(multilinear_evaluations_base_d, multilinear_evaluations_base, num_batches*sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(multilinear_evaluations_extension_d, multilinear_evaluations_extension, num_batches * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice));

    for(int i = 0; i < round_idx; i++) {
        //print_debug<<<1, 1>>>(multilinear_evaluations_extension_d, ((1 << (n-1)) / 32) >> i);
        //check(cudaDeviceSynchronize());
            fold_large_list_halves<<<8192, 256>>>(
                multilinear_evaluations_extension_d,
                multilinear_evaluations_extension_d,
                random_challenge_d + BITS_WIDTH * i,
                /*((1 << (n-1)) / 32) >> i,
                ((1 << n) / 32) >> i,
                ((1 << n) / 32) >> i,*/
                (1 << (n-1)) >> i >> 5,  
                (1 << (n)), 
                (1 << (n)), 
                1
            );
        check(cudaDeviceSynchronize());
    }

    //int j = 0;
    for(int j = 0; j < ((1 << n) / 32) >> round_idx; j++) {
        uint32_t *get_batch_destination, *fold_batch_destination;
        uint32_t *get_batch_destination_upper;
        uint32_t *get_batch_destination_lower;

        get_batch_destination = (uint32_t*) malloc(BITS_WIDTH * sizeof(uint32_t));
        fold_batch_destination = (uint32_t*) malloc(BITS_WIDTH * sizeof(uint32_t));

        get_batch_full(
            multilinear_evaluations_base,
            random_challenge_batch,
            get_batch_destination,
            j,
            round_idx,
            n,
            1
        );

        check(cudaMemcpy(fold_batch_destination, multilinear_evaluations_extension_d + j*BITS_WIDTH, BITS_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for(int i = 0; i < BITS_WIDTH; i++) {
            REQUIRE(get_batch_destination[i] == fold_batch_destination[i]);
        }
    }
}

TEST_CASE("test_fold_multiple") {
    const uint32_t n = 28;
    const uint32_t d = 2;
    const uint32_t num_batches = (1 << n) / 32;
    const uint32_t round_idx = 7; // works only for 1
    
    std::srand(0);

    uint32_t* multilinear_evaluations_p1 = (uint32_t*) malloc(num_batches * BITS_WIDTH * sizeof(uint32_t));
    uint32_t* multilinear_evaluations_base = (uint32_t*) malloc((d-1) * num_batches * sizeof(uint32_t));

    uint32_t* multilinear_evaluations_extension = (uint32_t*) malloc(d * num_batches * BITS_WIDTH * sizeof(uint32_t));
    uint32_t* random_challenge = (uint32_t*) malloc(round_idx * INTS_PER_VALUE * sizeof(uint32_t));
    uint32_t* random_challenge_batch = (uint32_t*) malloc(round_idx * BITS_WIDTH * sizeof(uint32_t));


    uint32_t* multilinear_evaluations_p1_d;
    uint32_t* multilinear_evaluations_base_d;
    uint32_t* multilinear_evaluations_extension_d;
    uint32_t* multilinear_evaluations_folded_d;
    uint32_t* random_challenge_d;

    check(cudaMalloc(&multilinear_evaluations_p1_d, num_batches * BITS_WIDTH * sizeof(uint32_t)));
    check(cudaMalloc(&multilinear_evaluations_base_d, (d-1) * num_batches * sizeof(uint32_t)));
    check(cudaMalloc(&multilinear_evaluations_extension_d, d * num_batches * BITS_WIDTH * sizeof(uint32_t)));
    check(cudaMalloc(&multilinear_evaluations_folded_d, d * num_batches * BITS_WIDTH * sizeof(uint32_t)));
    check(cudaMalloc(&random_challenge_d, round_idx * BITS_WIDTH * sizeof(uint32_t)));

    for(int j = 0; j < round_idx; j++) {
        for(int i = 0; i < INTS_PER_VALUE; i++) {
            random_challenge[i + j*INTS_PER_VALUE] = std::rand();
        }
        BitsliceUtils<BITS_WIDTH>::repeat_value_bitsliced(random_challenge_batch + j*BITS_WIDTH, random_challenge + j * INTS_PER_VALUE);
    }

    check(cudaMemcpy(random_challenge_d, random_challenge_batch, round_idx * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice));

    for(int i = 0; i < num_batches; i++) {
        for(int j = 0; j < BITS_WIDTH; j++) {
            uint32_t batch = std::rand();
            multilinear_evaluations_p1[i*BITS_WIDTH + j] = batch;
        }
    }
    check(cudaMemcpy(multilinear_evaluations_p1_d, multilinear_evaluations_p1, num_batches * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(multilinear_evaluations_extension, multilinear_evaluations_p1, num_batches * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyHostToHost));

    for(int j = 0; j < (d-1); j++) {
        for(int i = 0; i < num_batches; i++) {
            uint32_t batch = std::rand();
            multilinear_evaluations_base[i] = batch;
            multilinear_evaluations_extension[BITS_WIDTH * num_batches * (j+1) + BITS_WIDTH * i] = batch;
        }
    }
    check(cudaMemcpy(multilinear_evaluations_base_d, multilinear_evaluations_base, (d-1) * num_batches*sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(multilinear_evaluations_extension_d, multilinear_evaluations_extension, d * num_batches * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice));

    printf("%u %u\n", multilinear_evaluations_extension[num_batches * BITS_WIDTH], multilinear_evaluations_extension[num_batches * BITS_WIDTH + num_batches * BITS_WIDTH / 2]);

    uint32_t test_folded_batch[BITS_WIDTH];
    fold_batch(
        multilinear_evaluations_extension + num_batches * BITS_WIDTH,
        multilinear_evaluations_extension + num_batches * BITS_WIDTH + num_batches * BITS_WIDTH / 2,
        test_folded_batch,
        random_challenge_batch,
        false
    );
    printf("%u\n", test_folded_batch[0]);

    for(int i = 0; i < round_idx; i++) {
        fold_large_list_halves<<<8192, 256>>>(
            multilinear_evaluations_extension_d,
            multilinear_evaluations_extension_d,
            random_challenge_d + BITS_WIDTH * i,
            ((1 << (n-1)) / 32) >> i,
            1 << n,
            1 << n,
            d
        );
        check(cudaDeviceSynchronize());
    }

    fold_multiple(
        multilinear_evaluations_extension_d,
        multilinear_evaluations_base_d,
        random_challenge_batch,
        multilinear_evaluations_folded_d,
        round_idx,
        n,
        d
    );

    uint32_t num_batches_folded = d * ((1 << n) / 32);

    uint32_t* multilinear_evaluations_folded = (uint32_t*) malloc(num_batches_folded * BITS_WIDTH * sizeof(uint32_t));
    uint32_t* multilinear_evaluations_folded_multiple = (uint32_t*) malloc(num_batches_folded * BITS_WIDTH * sizeof(uint32_t));

    cudaMemcpy(multilinear_evaluations_folded, multilinear_evaluations_extension_d, num_batches_folded * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(multilinear_evaluations_folded_multiple, multilinear_evaluations_folded_d, num_batches_folded * BITS_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for(int j = 1; j < d; j++) {
        for(int i = 0; i < ((1 << (n-round_idx)) / 32) * BITS_WIDTH; i++) {
            if(multilinear_evaluations_folded[j * num_batches * BITS_WIDTH + i] != multilinear_evaluations_folded_multiple[j * num_batches * BITS_WIDTH + i]) {
                printf("%u %u %u\n", i, j, j * num_batches * BITS_WIDTH + i);
            }
            REQUIRE(multilinear_evaluations_folded[j * num_batches * BITS_WIDTH + i] == multilinear_evaluations_folded_multiple[j * num_batches * BITS_WIDTH + i]);
        }
    }
}