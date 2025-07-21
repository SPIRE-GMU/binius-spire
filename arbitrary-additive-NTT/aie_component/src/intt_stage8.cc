#include <adf.h>
#include "adf/window/types.h"
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

//from now on we use for loop for generalizaton
void intt_stage8(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output)
{
    uint32_t a[128],b[128];
    
    
    for(int i=0;i<2048/256;i++)
    chess_prepare_for_pipelining
    chess_loop_range(32,)    
    {
    //    a[0] = readi 
        for(int j=0 ;j<128;j++)chess_prepare_for_pipelining
            chess_loop_range(32,) {
            a[j] = readincr(cb_input);
        }
        for(int k=0 ;k<128;k++)chess_prepare_for_pipelining
            chess_loop_range(32,) {
            b[k] = readincr(cb_input);
        }
        
        for(int u=0;u<128;u+=2)chess_prepare_for_pipelining
            chess_loop_range(32,){
            exchange1(&a[u],&b[u]);  
            exchange2(&a[u+1],&b[u+1]);            
        }
   
        
        for(int j=0 ;j<128;j++)chess_prepare_for_pipelining
            chess_loop_range(32,) {
            writeincr(cb_output,a[j]);
        }
        for(int j=0 ;j<128;j++)chess_prepare_for_pipelining
            chess_loop_range(32,) {
            writeincr(cb_output,b[j]);
        }
    }       
   
}

