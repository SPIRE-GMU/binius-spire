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
    
    for(int i=0;i<1024;i++){
                    
        a[i]=readincr(input);
        // printf("read %d from PL %d\n",i,res);
        // writeincr(outputw,res);  
    }
    
    for(int i=0;i<1024;i++){                    
        b[i]=readincr(input);
         
    }   

    for(int j=0;j<1024/2;j+=2){
        exchange1(&a[j],&b[j]);
        exchange2(&a[j+1],&b[j+1]); // two instruction arrays
        
    }
    for(int j=0;j<1024;j++){
           writeincr(outputw,a[j]);         
    }
    for(int j=0;j<1024;j++){
           writeincr(outputw,b[j]);         
    }
    
}

