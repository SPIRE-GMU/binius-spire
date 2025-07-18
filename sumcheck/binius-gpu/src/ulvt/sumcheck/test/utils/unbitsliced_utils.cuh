#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include "../../../finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"

#ifndef UNBITSLICED_UTILS
#define UNBITSLICED_UTILS

void uint128_to_hex(__uint128_t value, char *buffer) {
    // Temporary buffer to hold hex digits (max 32 hex digits for 128 bits) + 1 for '\0'
    char temp[33] = {0};
    int i = 32;
    temp[i] = '\0';

    if (value == 0) {
        // Special case for 0
        buffer[0] = '0';
        buffer[1] = '\0';
        return;
    }

    while (value > 0 && i > 0) {
        --i;
        uint8_t digit = value & 0xF;
        temp[i] = (digit < 10) ? ('0' + digit) : ('a' + (digit - 10));
        value >>= 4;
    }

    // Copy result to output buffer
    int j = 0;
    while (temp[i]) {
        buffer[j++] = temp[i++];
    }
    buffer[j] = '\0';
}

void print_uint128_hex(__uint128_t value) {
    uint64_t high = static_cast<uint64_t>(value >> 64);
    uint64_t low = static_cast<uint64_t>(value & 0xFFFFFFFFFFFFFFFF);

    if (high) {
        std::cout << std::hex << "0x" << high
                  << std::setw(16) << std::setfill('0') << low << std::dec;
    } else {
        std::cout << std::hex << "0x" << low << std::dec;
    }
}

void mult128_unbitsliced_no_limb(__uint128_t a, __uint128_t b, __uint128_t& result) {
    uint32_t a_batch[128];
    uint32_t b_batch[128];
    uint32_t result_batch[128];
    //printf("mult %u %u\n", a[0], b[0]);
    for(int i = 0; i < 128; i++) {
        //int j = 128 - i;
        //printf("limb_idx %d bit_idx %d a[limb_idx] %u\n", limb_idx, bit_idx, a[limb_idx]);
        if(a & ((__uint128_t) 1 << i)) {
            a_batch[i] = 0xFFFFFFFF;
        } else {
            a_batch[i] = 0;
        }
        if(b & ((__uint128_t) 1 << i)) {
            b_batch[i] = 0xFFFFFFFF;
        } else {
            b_batch[i] = 0;
        }
        result_batch[i] = 0;
    } 

    result = 0;


    multiply_unrolled<7>(a_batch, b_batch, result_batch);
    for(int i = 0; i < 128; i++) {
        //result ^= (((__uint128_t) result_batch[i]) & 1) << i;
        __uint128_t bit = ((__uint128_t) 1) << i;
        result ^= bit * (result_batch[i] & 1);
    }
}

void mult128_unbitsliced(uint32_t* a, uint32_t* b, uint32_t* result) {
    uint32_t a_batch[128];
    uint32_t b_batch[128];
    uint32_t result_batch[128];
    //printf("mult %u %u\n", a[0], b[0]);
    for(int i = 0; i < 128; i++) {
        //int j = 128 - i;
        int limb_idx = i / 32;
        int bit_idx = i % 32;
        //printf("limb_idx %d bit_idx %d a[limb_idx] %u\n", limb_idx, bit_idx, a[limb_idx]);
        if(a[limb_idx] & (1 << bit_idx)) {
            a_batch[i] = 0xFFFFFFFF;
        } else {
            a_batch[i] = 0;
        }
        if(b[limb_idx] & (1 << bit_idx)) {
            b_batch[i] = 0xFFFFFFFF;
        } else {
            b_batch[i] = 0;
        }
        result_batch[i] = 0;
    } 

    for(int i = 0; i < 4; i++) result[i] = 0;

    //__uint128_t res128 = 0;

    multiply_unrolled<7>(a_batch, b_batch, result_batch);
    for(int i = 0; i < 128; i++) {
        int limb_idx = i / 32;
        int bit_idx = i % 32;
        result[limb_idx] ^= (result_batch[i] & 1) << bit_idx;
        //res128 += ((__uint128_t) 1 << i) * (result_batch[i] & 1);
    }

    /*char res128s[33];
    uint128_to_hex(res128, res128s);
    printf("%s\n", res128s);

    for(int i = 0; i < 4; i++) 
        printf("%x", result[i]);
    printf("\n");*/
}

void untranspose128(const uint32_t* bitsliced, int num_batches, uint32_t* output) {
    const int num_bits = 128;         // each number is 128-bit
    const int integers_per_batch = 32;
    const int limbs_per_integer = 4;

    // Zero out output
    memset(output, 0, num_batches * integers_per_batch * limbs_per_integer * sizeof(uint32_t));

    for (int bit = 0; bit < num_bits; ++bit) {
        int limb_index = bit / 32;
        int bit_in_limb = bit % 32;

        for (int batch = 0; batch < num_batches; ++batch) {
            uint32_t slice = bitsliced[batch * num_batches + bit];

            for (int i = 0; i < 32; ++i) {
                int index = (batch * integers_per_batch + i) * limbs_per_integer + limb_index;
                if ((slice >> i) & 1) {
                    output[index] |= (1u << bit_in_limb);
                }
            }
        }
    }
    /*printf("%d\n", num_batches*BITS_WIDTH);
    memset(output, 0, num_batches * 32 * 4 * sizeof(uint32_t));
    for (int batch = 0; batch < num_batches; batch++) {
        for (int bit = 0; bit < BITS_WIDTH; bit++) {
            int limb_idx = bit / 32;
            int bit_idx = bit % 32;
            for(int i = 0; i < 32; i++) {
                int idx = batch * 32;
                if(bitsliced[batch * BITS_WIDTH + bit] & (1 << i)) {
                    //printf("output[%d] out of %d\n", INTS_PER_VALUE * idx + limb_idx, INTS_PER_VALUE*num_batches*32);
                    output[INTS_PER_VALUE * idx + limb_idx] |= (1 << bit_idx);
                }
            }
        }
    }*/
}
#endif

