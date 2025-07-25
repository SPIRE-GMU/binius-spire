#include <adf.h>
#include "adf/window/types.h"
#include "kernels.h"
#include "adf/intrinsics.h"
#include "adf/stream/types.h"
#include "aie_api/aie.hpp"
// #include "adf/x86sim/streamApi.h"
// #include "include.h"



//from now on we use for loop for generalizaton
void intt_stage4_opt(       input_stream_int32 * cb_input,output_stream_int32 * cb_output)
{
    uint32_t a[8],b[8];
    
    uint16_t low_a0,high_a0,low_b0,high_b0;
    uint16_t low_a1,high_a1,low_b1,high_b1;
    for(int i=0;i<2048/16;i++)
    chess_prepare_for_pipelining
    chess_loop_range(32,)    
    {
    //  // --- 读入 a[0~7] ---
        a[0] = readincr(cb_input);
        a[1] = readincr(cb_input);
        a[2] = readincr(cb_input);
        a[3] = readincr(cb_input);
        a[4] = readincr(cb_input);
        a[5] = readincr(cb_input);
        a[6] = readincr(cb_input);
        a[7] = readincr(cb_input);

        // --- b[0] 和 a[0] 交换 ---
        b[0] = readincr(cb_input);
        {
            uint16_t la = a[0] & 0xFFFF;
            uint16_t lb = b[0] & 0xFFFF;
            uint16_t hb = (b[0] >> 16) & 0xFFFF;

            a[0] = (a[0] & 0xFFFF0000) | hb;
            b[0] = ((uint32_t)la << 16) | lb;
        }

        // --- b[1] 和 a[1] 交换 ---
        b[1] = readincr(cb_input);
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

        // --- b[2] 和 a[2] 交换 ---
        b[2] = readincr(cb_input);
        {
            uint16_t la = a[2] & 0xFFFF;
            uint16_t lb = b[2] & 0xFFFF;
            uint16_t hb = (b[2] >> 16) & 0xFFFF;

            a[2] = (a[2] & 0xFFFF0000) | hb;
            b[2] = ((uint32_t)la << 16) | lb;
        }

        b[3] = readincr(cb_input);
        {
            uint16_t la = a[3] & 0xFFFF;
            uint16_t lb = b[3] & 0xFFFF;
            uint16_t hb = (b[3] >> 16) & 0xFFFF;

            a[3] = (a[3] & 0xFFFF0000) | hb;
            b[3] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[2]);
        writeincr(cb_output, a[3]);

        b[4] = readincr(cb_input);
        {
            uint16_t la = a[4] & 0xFFFF;
            uint16_t lb = b[4] & 0xFFFF;
            uint16_t hb = (b[4] >> 16) & 0xFFFF;

            a[4] = (a[4] & 0xFFFF0000) | hb;
            b[4] = ((uint32_t)la << 16) | lb;
        }

        b[5] = readincr(cb_input);
        {
            uint16_t la = a[5] & 0xFFFF;
            uint16_t lb = b[5] & 0xFFFF;
            uint16_t hb = (b[5] >> 16) & 0xFFFF;

            a[5] = (a[5] & 0xFFFF0000) | hb;
            b[5] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[4]);
        writeincr(cb_output, a[5]);

        b[6] = readincr(cb_input);
        {
            uint16_t la = a[6] & 0xFFFF;
            uint16_t lb = b[6] & 0xFFFF;
            uint16_t hb = (b[6] >> 16) & 0xFFFF;

            a[6] = (a[6] & 0xFFFF0000) | hb;
            b[6] = ((uint32_t)la << 16) | lb;
        }

        b[7] = readincr(cb_input);
        {
            uint16_t la = a[7] & 0xFFFF;
            uint16_t lb = b[7] & 0xFFFF;
            uint16_t hb = (b[7] >> 16) & 0xFFFF;

            a[7] = (a[7] & 0xFFFF0000) | hb;
            b[7] = ((uint32_t)la << 16) | lb;
        }

        writeincr(cb_output, a[6]);
        writeincr(cb_output, a[7]);

        // --- 最后输出 b[0~7] ---
        for (int k = 0; k < 8; k++) {
            writeincr(cb_output, b[k]);
        }
    }       
   
}

