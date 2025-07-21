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
void intt_stage0
(       
	input_stream_uint32 * cb_input,
	output_stream_uint32 * cb_output)
{
   uint32_t a,b,res;    
    uint16_t low,high;
    
    for(int i=0;i<2048;i++)
    chess_prepare_for_pipelining
    chess_loop_range(32,)    
    {
       a = readincr(cb_input);
       low = a &0xFFFF;
       high = (a>>16)&0xFFFF;
       res = ((uint32_t)low<< 16 ) | high; 
    //    printf("read %d from mm2s %d\n",i,a);
        writeincr(cb_output,res);
    }       
   
}

