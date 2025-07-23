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
//accept polynomial from PS, rearrange for binius_mul.  reverse high-low when carrying 16uint from 32uint channel
//for example, I prepare number [1,2,3,4,5,6,7,8] in 16uint, but actually receive [2,1,4,3,6,5,8,7] in 32uint
void intt_stage3_opt(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output)
{
    uint32_t a[4],b[4];
        // v4uint32 a,b;
    uint32_t res1,res2;    
    uint16_t low_a0,high_a0,low_b0,high_b0;
    uint16_t low_a1,high_a1,low_b1,high_b1;

    for(int i=0;i<2048/8;i++)
    chess_prepare_for_pipelining
    chess_loop_range(32,)    
    {
      
       
        a[0] = readincr(cb_input);
        a[1] = readincr(cb_input);
        a[2] = readincr(cb_input);
        a[3] = readincr(cb_input);

        b[0] = readincr(cb_input); 
            low_a0 = a[0] & 0xFFFF;
            low_b0 = b[0] & 0xFFFF;
            high_b0 = (b[0] >> 16) & 0xFFFF;
    
            a[0] = (a[0] & 0xFFFF0000) | high_b0;
            b[0]   = ((uint32_t)low_a0 << 16) | low_b0;



       b[1] = readincr(cb_input);          
            low_a1 = a[1] & 0xFFFF;
            low_b1 = b[1] & 0xFFFF;
            high_b1 = (b[1] >> 16) & 0xFFFF;
    
            a[1] = (a[1] & 0xFFFF0000) | high_b1;
            b[1]   = ((uint32_t)low_a1 << 16) | low_b1;
        writeincr(cb_output,a[0]);
        writeincr(cb_output,a[1]);
       
//        b[2] = readincr(cb_input);  
//        b[3] = readincr(cb_input);  
        
        b[2] = readincr(cb_input); 
            low_a0 = a[2] & 0xFFFF;
            low_b0 = b[2] & 0xFFFF;
            high_b0 = (b[2] >> 16) & 0xFFFF;
    
            a[2] = (a[2] & 0xFFFF0000) | high_b0;
            b[2]   = ((uint32_t)low_a0 << 16) | low_b0;



       b[3] = readincr(cb_input);          
            low_a1 = a[3] & 0xFFFF;
            low_b1 = b[3] & 0xFFFF;
            high_b1 = (b[3] >> 16) & 0xFFFF;
    
            a[3] = (a[3] & 0xFFFF0000) | high_b1;
            b[3]   = ((uint32_t)low_a1 << 16) | low_b1;


        writeincr(cb_output,a[2]);
        writeincr(cb_output,a[3]);
        writeincr(cb_output,b[0]);
        writeincr(cb_output,b[1]);
        
        writeincr(cb_output,b[2]);
        writeincr(cb_output,b[3]);
    }       
   
}

