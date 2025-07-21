#include <adf.h>
#include "adf/window/types.h"
#include "kernels.h"
#include "adf/intrinsics.h"
#include "adf/stream/types.h"
#include "aie_api/aie.hpp"
// #include "adf/x86sim/streamApi.h"
// #include "include.h"


//accept polynomial from PS, rearrange for binius_mul.  reverse high-low when carrying 16uint from 32uint channel
//for example, I prepare number [1,2,3,4,5,6,7,8] in 16uint, but actually receive [2,1,4,3,6,5,8,7] in 32uint
void intt_stage1
(       
	input_stream_uint32 * cb_input,
	output_stream_uint32 * cb_output)
{
    uint32_t a,b,res1,res2;    
    uint16_t low_a,high_a,low_b,high_b;
    
    for(int i=0;i<2048/2;i++)
    chess_prepare_for_pipelining
    chess_loop_range(32,)    
    {
       a = readincr(cb_input);
       b = readincr(cb_input);    
        // printf("A [%d]read =%d",i,a);
        // printf("B [%d]read =%d",i,b);
       low_a = a & 0xFFFF;
       low_b = b & 0xFFFF;
       high_b = (b>>16)&0xFFFF;
       res1 = (a & 0xFFFF0000 ) | high_b; 
       res2 = ((uint32_t)low_a<< 16 ) | low_b; 
       
    //    res2 = (b & 0xFFFF0000 ) + low_a;
    //    printf("res1 %d = %d\n",i,res1);
        // printf("res2 %d = %d\n",i,res2);

        writeincr(cb_output,res1);
        writeincr(cb_output,res2);
    }       
   
}

