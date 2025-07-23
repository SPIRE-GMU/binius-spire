#include <adf.h>
#include "adf/window/types.h"
#include "kernels.h"
#include "adf/intrinsics.h"
#include "adf/stream/types.h"
#include "aie_api/aie.hpp"
// #include "adf/x86sim/streamApi.h"
// #include "include.h"


//from now on we use for loop for generalizaton
void intt_stage5_opt(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output)
{
    uint32_t a[16],b[16];
    
    uint16_t low_a0,high_a0,low_b0,high_b0;
    uint16_t low_a1,high_a1,low_b1,high_b1;
    
    
    for(int i=0;i<2048/32;i++)
    chess_prepare_for_pipelining
    chess_loop_range(64,)    
    {
    //===================================================//
    //    pipeline read  14us, no pipelined read 10us    //
    //===================================================//
        // for(int j=0;j<16;j++)chess_prepare_for_pipelining
        // chess_loop_range(64,) {
        //   a[j] = readincr(cb_input);
          
        // }
        // for(int j=0;j<16;j++)
        // chess_prepare_for_pipelining
        // chess_loop_range(64,) {
        //   b[j] = readincr(cb_input);
        // }
       //============================== //
        //     no pipelined read 10us   //
        //=============================//
        // for(int j=0;j<16;j++) {
        //   a[j] = readincr(cb_input);
          
        // }
        // for(int j=0;j<16;j++)
        //  {
        //   b[j] = readincr(cb_input);          
        // }
        
        //=======================================//
        //just let compiler optimize itself  8us //
        //=======================================//
        a[0] = readincr(cb_input);
        a[1] = readincr(cb_input);
        a[2] = readincr(cb_input);
        a[3] = readincr(cb_input);
        a[4] = readincr(cb_input);
        a[5] = readincr(cb_input);
        a[6] = readincr(cb_input);
        a[7] = readincr(cb_input);
        a[8] = readincr(cb_input);
        a[9] = readincr(cb_input);
        a[10] = readincr(cb_input);
        a[11] = readincr(cb_input);
        a[12] = readincr(cb_input);
        a[13] = readincr(cb_input);
        a[14] = readincr(cb_input);
        a[15] = readincr(cb_input);
        b[0] = readincr(cb_input);
        b[1] = readincr(cb_input);
        b[2] = readincr(cb_input);
        b[3] = readincr(cb_input);
        b[4] = readincr(cb_input);
        b[5] = readincr(cb_input);
        b[6] = readincr(cb_input);
        b[7] = readincr(cb_input);
        b[8] = readincr(cb_input);
        b[9] = readincr(cb_input);
        b[10] = readincr(cb_input);
        b[11] = readincr(cb_input);
        b[12] = readincr(cb_input);
        b[13] = readincr(cb_input);
        b[14] = readincr(cb_input);
        b[15] = readincr(cb_input);
         /// I have to unroll them out, to make sure they are implemented in pipeline.
        {
            uint16_t la  = a[0] & 0xFFFF;
            uint16_t lb = b[0] & 0xFFFF;
            uint16_t hb = (b[0] >> 16) & 0xFFFF;

            a[0] = (a[0] & 0xFFFF0000) | hb;
            b[0] = ((uint32_t)la << 16) | lb;
        }

        // --- b[1] 和 a[1] 交换 ---    

        {
             uint16_t la = a[1] & 0xFFFF;
             uint16_t lb = b[1] & 0xFFFF;
             uint16_t hb = (b[1] >> 16) & 0xFFFF;

            a[1] = (a[1] & 0xFFFF0000) | hb;
            b[1] = ((uint32_t)la << 16) | lb;
        }

        // --- 立刻输出 a[0], a[1] ---
        writeincr(cb_output, a[0]);
        writeincr(cb_output, a[1]);


        {
        
            uint16_t la = a[2] & 0xFFFF;
            uint16_t lb = b[2] & 0xFFFF;
            uint16_t hb = (b[2] >> 16) & 0xFFFF;

            a[2] = (a[2] & 0xFFFF0000) | hb;
            b[2] = ((uint32_t)la << 16) | lb;
        
        }

        {
            uint16_t la = a[3] & 0xFFFF;
            uint16_t lb = b[3] & 0xFFFF;
            uint16_t hb = (b[3] >> 16) & 0xFFFF;

            a[3] = (a[3] & 0xFFFF0000) | hb;
            b[3] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[2]);
        writeincr(cb_output, a[3]);


        {
            uint16_t la = a[4] & 0xFFFF;
            uint16_t lb = b[4] & 0xFFFF;
            uint16_t hb = (b[4] >> 16) & 0xFFFF;

            a[4] = (a[4] & 0xFFFF0000) | hb;
            b[4] = ((uint32_t)la << 16) | lb;
        }


        {
            uint16_t la = a[5] & 0xFFFF;
            uint16_t lb = b[5] & 0xFFFF;
            uint16_t hb = (b[5] >> 16) & 0xFFFF;

            a[5] = (a[5] & 0xFFFF0000) | hb;
            b[5] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[4]);
        writeincr(cb_output, a[5]);


        {
            uint16_t la = a[6] & 0xFFFF;
            uint16_t lb = b[6] & 0xFFFF;
            uint16_t hb = (b[6] >> 16) & 0xFFFF;

            a[6] = (a[6] & 0xFFFF0000) | hb;
            b[6] = ((uint32_t)la << 16) | lb;
        }


        {
            uint16_t la = a[7] & 0xFFFF;
            uint16_t lb = b[7] & 0xFFFF;
            uint16_t hb = (b[7] >> 16) & 0xFFFF;

            a[7] = (a[7] & 0xFFFF0000) | hb;
            b[7] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[6]);
        writeincr(cb_output, a[7]);
        

        {
            uint16_t la = a[8] & 0xFFFF;
            uint16_t lb = b[8] & 0xFFFF;
            uint16_t hb = (b[8] >> 16) & 0xFFFF;

            a[8] = (a[8] & 0xFFFF0000) | hb;
            b[8] = ((uint32_t)la << 16) | lb;
        }


        {
            uint16_t la = a[9] & 0xFFFF;
            uint16_t lb = b[9] & 0xFFFF;
            uint16_t hb = (b[9] >> 16) & 0xFFFF;

            a[9] = (a[9] & 0xFFFF0000) | hb;
            b[9] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[8]);
        writeincr(cb_output, a[9]);


        {
            uint16_t la = a[10] & 0xFFFF;
            uint16_t lb = b[10] & 0xFFFF;
            uint16_t hb = (b[10] >> 16) & 0xFFFF;

            a[10] = (a[10] & 0xFFFF0000) | hb;
            b[10] = ((uint32_t)la << 16) | lb;
        }


        {
            uint16_t la = a[11] & 0xFFFF;
            uint16_t lb = b[11] & 0xFFFF;
            uint16_t hb = (b[11] >> 16) & 0xFFFF;

            a[11] = (a[11] & 0xFFFF0000) | hb;
            b[11] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[10]);
        writeincr(cb_output, a[11]);


        {
            uint16_t la = a[12] & 0xFFFF;
            uint16_t lb = b[12] & 0xFFFF;
            uint16_t hb = (b[12] >> 16) & 0xFFFF;

            a[12] = (a[12] & 0xFFFF0000) | hb;
            b[12] = ((uint32_t)la << 16) | lb;
        }


        {
            uint16_t la = a[13] & 0xFFFF;
            uint16_t lb = b[13] & 0xFFFF;
            uint16_t hb = (b[13] >> 16) & 0xFFFF;

            a[13] = (a[13] & 0xFFFF0000) | hb;
            b[13] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[12]);
        writeincr(cb_output, a[13]);


        {
            uint16_t la = a[14] & 0xFFFF;
            uint16_t lb = b[14] & 0xFFFF;
            uint16_t hb = (b[14] >> 16) & 0xFFFF;

            a[14] = (a[14] & 0xFFFF0000) | hb;
            b[14] = ((uint32_t)la << 16) | lb;
        }


        {
            uint16_t la = a[15] & 0xFFFF;
            uint16_t lb = b[15] & 0xFFFF;
            uint16_t hb = (b[15] >> 16) & 0xFFFF;

            a[15] = (a[15] & 0xFFFF0000) | hb;
            b[15] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[14]);
        writeincr(cb_output, a[15]);



        for(int j=0 ;j<16;j++)chess_prepare_for_pipelining
            chess_loop_range(32,) {
            writeincr(cb_output,b[j]);
        }
    }       
   
}

