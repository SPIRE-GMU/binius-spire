#include <cassert>
#include <cstdio>
#include <cuda/std/utility>
#include "binary_mul.cu"


int main() {
    assert(BinaryTower<5>::multiply(3, 3) == 2);

    assert(BinaryTower<5>::add(2, 3) == 1);

    assert(BinaryTower<5>::square(1) == 1);

    assert(BinaryTower<5>::multiply(BinaryTower<5>::inverse(10),10) == 1);

    printf("All tests passed!\n");
    
    return 0;
}

