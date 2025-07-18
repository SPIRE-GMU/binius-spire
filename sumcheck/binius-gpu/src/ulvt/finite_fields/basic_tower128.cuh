#include <iostream>
#include <vector>
#include <algorithm> // For std::max

// Function to perform addition/subtraction in GF(2) (XOR operation)
void gf2_add(unsigned int* result, const unsigned int* a, const unsigned int* b, int num_limbs) {
    for (int i = 0; i < num_limbs; ++i) {
        result[i] = a[i] ^ b[i];
    }
}

// Function to shift a 128-bit number left by a specified number of bits
// result = value << shift_bits
void gf2_lshift(unsigned int* result, const unsigned int* value, int shift_bits, int num_limbs) {
    if (shift_bits == 0) {
        for (int i = 0; i < num_limbs; ++i) {
            result[i] = value[i];
        }
        return;
    }

    int limb_shift = shift_bits / 32;
    int bit_shift = shift_bits % 32;

    for (int i = num_limbs - 1; i >= 0; --i) {
        if (i - limb_shift >= 0) {
            result[i] = value[i - limb_shift] << bit_shift;
            if (bit_shift > 0 && i - limb_shift - 1 >= 0) {
                result[i] |= value[i - limb_shift - 1] >> (32 - bit_shift);
            }
        } else {
            result[i] = 0;
        }
    }
}

// Function to extract a "half" (lower or upper) from a 128-bit number
// This is equivalent to v & halfmask or v >> halflen in Python
void get_half(unsigned int* half_result, const unsigned int* full_value, int start_bit, int half_length_bits) {
    for (int i = 0; i < 4; ++i) { // Initialize result to 0
        half_result[i] = 0;
    }

    int start_limb = start_bit / 32;
    int start_bit_in_limb = start_bit % 32;

    int end_bit = start_bit + half_length_bits;
    int end_limb = (end_bit - 1) / 32; // -1 because it's inclusive
    int end_bit_in_limb = (end_bit - 1) % 32;

    for (int i = 0; i < (half_length_bits + 31) / 32; ++i) { // Iterate through the destination limbs
        int current_src_limb = start_limb + i;
        if (current_src_limb >= 4) break; // Should not happen for 128-bit numbers

        unsigned int lower_part = full_value[current_src_limb] >> start_bit_in_limb;
        unsigned int upper_part = 0;
        if (start_bit_in_limb > 0 && current_src_limb + 1 < 4) {
            upper_part = full_value[current_src_limb + 1] << (32 - start_bit_in_limb);
        }
        half_result[i] = lower_part | upper_part;

        // Mask out bits beyond half_length_bits in the last limb if necessary
        if (i == (half_length_bits + 31) / 32 - 1 && half_length_bits % 32 != 0) {
            unsigned int mask = (1U << (half_length_bits % 32)) - 1;
            half_result[i] &= mask;
        }
    }
}


// Helper function to check if a 128-bit number is zero
bool is_zero(const unsigned int* v, int num_limbs) {
    for (int i = 0; i < num_limbs; ++i) {
        if (v[i] != 0) {
            return false;
        }
    }
    return true;
}

// Helper function to check if a 128-bit number is equal to 1
bool is_one(const unsigned int* v, int num_limbs) {
    if (v[0] != 1) return false;
    for (int i = 1; i < num_limbs; ++i) {
        if (v[i] != 0) return false;
    }
    return true;
}

// Helper function to check if a 128-bit number is equal to (1 << quarterlen)
void set_power_of_2(unsigned int* result, int power, int num_limbs) {
    for (int i = 0; i < num_limbs; ++i) {
        result[i] = 0;
    }
    if (power < 0) return; // Invalid power

    int limb_index = power / 32;
    int bit_index = power % 32;

    if (limb_index < num_limbs) {
        result[limb_index] = (1U << bit_index);
    }
}

// Helper function to compare two 128-bit numbers
bool equals(const unsigned int* a, const unsigned int* b, int num_limbs) {
    for (int i = 0; i < num_limbs; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// Main binary multiplication function
// v1, v2: arrays of four 32-bit unsigned integers representing 128-bit numbers
// result: array of four 32-bit unsigned integers to store the 128-bit result
void binmul_128(const unsigned int* v1, const unsigned int* v2, unsigned int* result) {
    // Base cases for recursion
    // In GF(2), v1*v2 means bitwise AND for single bits, but for the recursive
    // approach, small values are handled as the multiplication itself.
    // The original Python code had `if v1 < 2 or v2 < 2: return v1 * v2`
    // For 128-bit numbers, this effectively means if either is 0 or 1.
    unsigned int temp_v1_val[4];
    unsigned int temp_v2_val[4];

    if (is_zero(v1, 4)) {
        for (int i = 0; i < 4; ++i) result[i] = 0;
        return;
    }
    if (is_zero(v2, 4)) {
        for (int i = 0; i < 4; ++i) result[i] = 0;
        return;
    }
    if (is_one(v1, 4)) {
        for (int i = 0; i < 4; ++i) result[i] = v2[i];
        return;
    }
    if (is_one(v2, 4)) {
        for (int i = 0; i < 4; ++i) result[i] = v1[i];
        return;
    }

    const int length = 128; // Fixed for 128-bit numbers
    const int halflen = length / 2; // 64
    const int quarterlen = length / 4; // 32

    unsigned int L1[4], R1[4], L2[4], R2[4];

    // L1, R1 = v1 & halfmask, v1 >> halflen
    // For 128-bit, halflen is 64. So L1 is the lower 64 bits, R1 is the upper 64 bits.
    L1[0] = v1[0]; L1[1] = v1[1]; L1[2] = 0; L1[3] = 0;
    R1[0] = v1[2]; R1[1] = v1[3]; R1[2] = 0; R1[3] = 0;

    // L2, R2 = v2 & halfmask, v2 >> halflen
    L2[0] = v2[0]; L2[1] = v2[1]; L2[2] = 0; L2[3] = 0;
    R2[0] = v2[2]; R2[1] = v2[3]; R2[2] = 0; R2[3] = 0;

    // Optimized special case (used to compute R1R2_high)
    // if (L1, R1) == (0, 1):
    // For 128-bit, (0, 1) means L1 is all zeros, and R1 is 1 (i.e., v1 is 2^64)
    unsigned int R1_val_is_1[4] = {1, 0, 0, 0};
    if (is_zero(L1, 4) && equals(R1, R1_val_is_1, 4)) { // R1 is [1, 0, 0, 0] when considering as 64-bit number
        unsigned int outR[4];
        unsigned int one_qlen[4];
        set_power_of_2(one_qlen, quarterlen, 4); // 1 << quarterlen (i.e., 2^32)
        binmul_128(outR, one_qlen, R2); // R2 is effectively 64-bit here
        
        unsigned int temp_outR_xor_L2[4];
        gf2_add(temp_outR_xor_L2, outR, L2, 4); // outR ^ L2

        unsigned int R2_shifted[4];
        gf2_lshift(R2_shifted, R2, halflen, 4); // R2 << halflen

        gf2_add(result, R2_shifted, temp_outR_xor_L2, 4);
        return;
    }

    unsigned int L1L2[4], R1R2[4], Z3[4];
    unsigned int R1R2_high[4];

    // L1L2 = binmul(L1, L2, halflen)
    binmul_128(L1L2, L1, L2);

    // R1R2 = binmul(R1, R2, halflen)
    binmul_128(R1R2, R1, R2);

    // R1R2_high = binmul(1 << quarterlen, R1R2, halflen)
    unsigned int one_qlen[4];
    set_power_of_2(one_qlen, quarterlen, 4); // 1 << quarterlen (i.e., 2^32)
    binmul_128(R1R2_high, one_qlen, R1R2);

    // Z3 = binmul(L1 ^ R1, L2 ^ R2, halflen)
    unsigned int L1_xor_R1[4], L2_xor_R2[4];
    gf2_add(L1_xor_R1, L1, R1, 4);
    gf2_add(L2_xor_R2, L2, R2, 4);
    binmul_128(Z3, L1_xor_R1, L2_xor_R2);

    // Final computation: L1L2 ^ R1R2 ^ ((Z3 ^ L1L2 ^ R1R2 ^ R1R2_high) << halflen)
    unsigned int term1[4], term2[4], term3_inner[4], term3_shifted[4];

    gf2_add(term1, L1L2, R1R2, 4); // L1L2 ^ R1R2

    gf2_add(term3_inner, Z3, L1L2, 4); // Z3 ^ L1L2
    gf2_add(term3_inner, term3_inner, R1R2, 4); // (Z3 ^ L1L2) ^ R1R2
    gf2_add(term3_inner, term3_inner, R1R2_high, 4); // (Z3 ^ L1L2 ^ R1R2) ^ R1R2_high

    gf2_lshift(term3_shifted, term3_inner, halflen, 4); // term3_inner << halflen

    gf2_add(result, term1, term3_shifted, 4); // (L1L2 ^ R1R2) ^ term3_shifted
}