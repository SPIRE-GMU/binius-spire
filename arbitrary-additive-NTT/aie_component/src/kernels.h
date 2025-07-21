#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <adf/window/types.h>
#include <adf/stream/types.h>

 void rearrange_out(
    input_stream_uint32 * input,
    output_stream_uint32 * outputw
)  ;

void intt_stage0(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage1(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage2(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage3(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage4(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage5(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage6(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage7(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage8(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage9(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage10(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;
void intt_stage11(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output) ;

#endif /* __KERNELS_H__ */

