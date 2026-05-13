#include "m31.cuh"
#include "cm31.cuh"

#include <iostream>
#include <cstdint>
#include <cstdlib>
// include random
#include <random>
#include "qm31.cuh"
using namespace std;

int main() {

    uint32_t P = M31::P;

    QM31 qm0(CM31(M31((uint32_t)P - 4743832), M31((uint32_t)4442)), CM31(M31((uint32_t)P - 5675), M31((uint32_t)P - 199938)));
    QM31 qm1(CM31(M31((uint32_t)P - 1560000), M31((uint32_t)P - 400000)), CM31(M31((uint32_t)P - 140000), M31((uint32_t)1000000)));

    std::cout << qm0.to_string() << std::endl;
    std::cout << qm1.to_string() << std::endl;
    std::cout << (qm0 * qm1).to_string() << std::endl;

    return 0;
}