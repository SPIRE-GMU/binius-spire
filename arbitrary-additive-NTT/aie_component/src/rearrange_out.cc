#include <adf.h>
#include "kernels.h"
#include "adf/intrinsics.h"
#include "adf/stream/types.h"
#include "aie_api/aie.hpp"
// #include "adf/x86sim/streamApi.h"
// #include "include.h"

inline void exchange1(uint32_t *a, uint32_t *b){
        uint16_t low_a,low_b,high_b;
        
        low_a = *a & 0xFFFF;
        low_b = *b & 0xFFFF;
        high_b = (*b>>16)&0xFFFF;

        *a =   ((*a) & 0xFFFF0000 ) | high_b; 
        *b = ((uint32_t)low_a<< 16 ) | low_b;           
};

inline void exchange2(uint32_t *a, uint32_t *b){
        uint16_t low_a,low_b,high_b;
        
        low_a = *a & 0xFFFF;
        low_b = *b & 0xFFFF;
        high_b = (*b>>16)&0xFFFF;

        *a =   ((*a) & 0xFFFF0000 ) | high_b; 
        *b = ((uint32_t)low_a<< 16 ) | low_b;           
};

void rearrange_out
(
    input_stream_uint32 * input,
    output_stream_uint32 * outputw
) {
    uint32_t a[1024],b[1024];
    uint32_t res;
    uint16_t low_a0,high_a0,low_b0,high_b0;
    uint16_t low_a1,high_a1,low_b1,high_b1;

    for(int i=0;i<1024;i++){
                    
        a[i]=readincr(input);
        // printf("read %d from PL %d\n",i,res);
        // writeincr(outputw,res);  
    }
    
    for(int j=0;j<1024;j+=2)
        chess_prepare_for_pipelining
        chess_loop_range(32,){

        
            b[j] = readincr(input); 
                low_a0 = a[j] & 0xFFFF;
                low_b0 = b[j] & 0xFFFF;
                high_b0 = (b[j] >> 16) & 0xFFFF;

                a[j] = (a[j] & 0xFFFF0000) | high_b0;
                b[j]   = ((uint32_t)low_a0 << 16) | low_b0;



            b[j+1] = readincr(input);          
                low_a1 = a[j+1] & 0xFFFF;
                low_b1 = b[j+1] & 0xFFFF;
                high_b1 = (b[j+1] >> 16) & 0xFFFF;

                a[j+1] = (a[j+1] & 0xFFFF0000) | high_b1;
                b[j+1]   = ((uint32_t)low_a1 << 16) | low_b1;
            writeincr(outputw,a[j]);
            writeincr(outputw,a[j+1]);
        }

        for(int k=0;k<1024;k++){
            writeincr(outputw,b[k]);
        }
    
}

