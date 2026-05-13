#pragma once
#include <cstdint>
#include <iostream>
#include "m31.cuh"

class CM31 {
public:
    static constexpr uint32_t BITS = 31;

    static constexpr uint32_t P = ((uint32_t) 1<<BITS) - 1;

    M31 subfield_elements[2];

    __host__ __device__ constexpr CM31() noexcept: subfield_elements{M31(), M31()} {}

    __host__ __device__ constexpr CM31(uint32_t val) noexcept: subfield_elements{M31(val), M31()} {}

    __host__ __device__ constexpr CM31(uint64_t val[2]) noexcept: subfield_elements{M31(val[0]), M31(val[1])} {}

    __host__ __device__ constexpr CM31(M31 lo, M31 hi) noexcept: subfield_elements{lo, hi} {}

    __host__ __device__ constexpr CM31 operator+(CM31 rhs) const { 
        return CM31(
            subfield_elements[0] + rhs.subfield_elements[0],
            subfield_elements[1] + rhs.subfield_elements[1]
        );
     }

    __host__ __device__ constexpr CM31 operator-(CM31 rhs) const { 
        return CM31(
            subfield_elements[0] - rhs.subfield_elements[0],
            subfield_elements[1] - rhs.subfield_elements[1]
        );
     }

    __host__ __device__ constexpr CM31& operator+=(CM31 rhs) {
        subfield_elements[0] += rhs.subfield_elements[0];
        subfield_elements[1] += rhs.subfield_elements[1];
        return *this;
     }

     __host__ __device__ constexpr CM31& operator-=(CM31 rhs) {
        subfield_elements[0] -= rhs.subfield_elements[0];
        subfield_elements[1] -= rhs.subfield_elements[1];
        return *this;
     }
    
    __host__ __device__ constexpr CM31 operator*(CM31 rhs) const { 
        return CM31(
            subfield_elements[0] * rhs.subfield_elements[0] - subfield_elements[1] * rhs.subfield_elements[1],
            subfield_elements[0] * rhs.subfield_elements[1] + subfield_elements[1] * rhs.subfield_elements[0]
        );
     }

    __host__ __device__ constexpr CM31& operator*=(CM31 rhs) {
        *this = *this * rhs;
        return *this;
     }

    __host__ __device__ constexpr bool operator==(CM31 rhs) const { 
        return subfield_elements[0] == rhs.subfield_elements[0] && subfield_elements[1] == rhs.subfield_elements[1];
     }

    __host__ __device__ constexpr bool operator!=(CM31 rhs) const { 
        return subfield_elements[0] != rhs.subfield_elements[0] || subfield_elements[1] != rhs.subfield_elements[1];
     }

     __host__ __device__ void write_to_u64(uint64_t dst[2]) const {
        subfield_elements[0].write_to_u64(&dst[0]);
        subfield_elements[1].write_to_u64(&dst[1]);
     }

     __host__ __device__ void sum_into_u64(uint64_t dst[2]) const {
        subfield_elements[0].sum_into_u64(&dst[0]);
        subfield_elements[1].sum_into_u64(&dst[1]);
     }

     __host__ std::string to_string() const {
        return "(" + subfield_elements[0].to_string() + ", " + subfield_elements[1].to_string() + ")";
     }
};