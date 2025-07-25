#include <adf.h>
#include "adf/window/types.h"
#include "kernels.h"
#include "adf/intrinsics.h"
#include "adf/stream/types.h"
#include "aie_api/aie.hpp"
// #include "adf/x86sim/streamApi.h"
// #include "include.h"



//from now on we use for loop for generalizaton
void intt_stage7_opt(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output)
{
    uint32_t a[64],b[64];
    
    
    for(int i=0;i<2048/128;i++)
    chess_prepare_for_pipelining
    chess_loop_range(32,)    
    {
    
        //==============================//
        //  looks clean but takes 14 us // 
        //==============================//
        // for(int j=0;j<64;j++) {
        //   a[j] = readincr(cb_input);
          
        // }
        // for(int j=0;j<64;j++)
        //  {
        //   b[j] = readincr(cb_input);          
        // }
        // for (int i = 0; i < 64; i += 2) chess_prepare_for_pipelining
        //     chess_loop_range(32,)
        //     {
        //         // --- 处理 a[i], b[i] ---
        //         {
        //             uint16_t la = a[i] & 0xFFFF;
        //             uint16_t lb = b[i] & 0xFFFF;
        //             uint16_t hb = (b[i] >> 16) & 0xFFFF;

        //             a[i] = (a[i] & 0xFFFF0000) | hb;
        //             b[i] = ((uint32_t)la << 16) | lb;
        //         }

        //         // --- 处理 a[i+1], b[i+1] ---
        //         {
        //             uint16_t la = a[i+1] & 0xFFFF;
        //             uint16_t lb = b[i+1] & 0xFFFF;
        //             uint16_t hb = (b[i+1] >> 16) & 0xFFFF;

        //             a[i+1] = (a[i+1] & 0xFFFF0000) | hb;
        //             b[i+1] = ((uint32_t)la << 16) | lb;
        //         }

        //         // --- 立刻输出处理后的 a[i], a[i+1] ---
        //         writeincr(cb_output, a[i]);
        //         writeincr(cb_output, a[i+1]);
        //     }


        //======================================//
        // flaten and let compiler optimize 8us
        //=====================================// 
    //     a[0] = readincr(cb_input);
    //     a[1] = readincr(cb_input);
    //     a[2] = readincr(cb_input);
    //     a[3] = readincr(cb_input);
    //     a[4] = readincr(cb_input);
    //     a[5] = readincr(cb_input);
    //     a[6] = readincr(cb_input);
    //     a[7] = readincr(cb_input);
    //     a[8] = readincr(cb_input);
    //     a[9] = readincr(cb_input);
    //     a[10] = readincr(cb_input);
    //     a[11] = readincr(cb_input);
    //     a[12] = readincr(cb_input);
    //     a[13] = readincr(cb_input);
    //     a[14] = readincr(cb_input);
    //     a[15] = readincr(cb_input);
    //     a[16] = readincr(cb_input);
    //     a[17] = readincr(cb_input);
    //     a[18] = readincr(cb_input);
    //     a[19] = readincr(cb_input);
    //     a[20] = readincr(cb_input);
    //     a[21] = readincr(cb_input);
    //     a[22] = readincr(cb_input);
    //     a[23] = readincr(cb_input);
    //     a[24] = readincr(cb_input);
    //     a[25] = readincr(cb_input);
    //     a[26] = readincr(cb_input);
    //     a[27] = readincr(cb_input);
    //     a[28] = readincr(cb_input);
    //     a[29] = readincr(cb_input);
    //     a[30] = readincr(cb_input);
    //     a[31] = readincr(cb_input);
    //     a[32+0] = readincr(cb_input);
    //     a[32+1] = readincr(cb_input);
    //     a[32+2] = readincr(cb_input);
    //     a[32+3] = readincr(cb_input);
    //     a[32+4] = readincr(cb_input);
    //     a[32+5] = readincr(cb_input);
    //     a[32+6] = readincr(cb_input);
    //     a[32+7] = readincr(cb_input);
    //     a[32+8] = readincr(cb_input);
    //     a[32+9] = readincr(cb_input);
    //     a[32+10] = readincr(cb_input);
    //     a[32+11] = readincr(cb_input);
    //     a[32+12] = readincr(cb_input);
    //     a[32+13] = readincr(cb_input);
    //     a[32+14] = readincr(cb_input);
    //     a[32+15] = readincr(cb_input);
    //     a[32+16] = readincr(cb_input);
    //     a[32+17] = readincr(cb_input);
    //     a[32+18] = readincr(cb_input);
    //     a[32+19] = readincr(cb_input);
    //     a[32+20] = readincr(cb_input);
    //     a[32+21] = readincr(cb_input);
    //     a[32+22] = readincr(cb_input);
    //     a[32+23] = readincr(cb_input);
    //     a[32+24] = readincr(cb_input);
    //     a[32+25] = readincr(cb_input);
    //     a[32+26] = readincr(cb_input);
    //     a[32+27] = readincr(cb_input);
    //     a[32+28] = readincr(cb_input);
    //     a[32+29] = readincr(cb_input);
    //     a[32+30] = readincr(cb_input);
    //     a[32+31] = readincr(cb_input);
    //     b[0] = readincr(cb_input);
    //     b[1] = readincr(cb_input);
    //     b[2] = readincr(cb_input);
    //     b[3] = readincr(cb_input);
    //     b[4] = readincr(cb_input);
    //     b[5] = readincr(cb_input);
    //     b[6] = readincr(cb_input);
    //     b[7] = readincr(cb_input);
    //     b[8] = readincr(cb_input);
    //     b[9] = readincr(cb_input);
    //     b[10] = readincr(cb_input);
    //     b[11] = readincr(cb_input);
    //     b[12] = readincr(cb_input);
    //     b[13] = readincr(cb_input);
    //     b[14] = readincr(cb_input);
    //     b[15] = readincr(cb_input);
    //     b[16] = readincr(cb_input);
    //     b[17] = readincr(cb_input);
    //     b[18] = readincr(cb_input);
    //     b[19] = readincr(cb_input);
    //     b[20] = readincr(cb_input);
    //     b[21] = readincr(cb_input);
    //     b[22] = readincr(cb_input);
    //     b[23] = readincr(cb_input);
    //     b[24] = readincr(cb_input);
    //     b[25] = readincr(cb_input);
    //     b[26] = readincr(cb_input);
    //     b[27] = readincr(cb_input);
    //     b[28] = readincr(cb_input);
    //     b[29] = readincr(cb_input);
    //     b[30] = readincr(cb_input);
    //     b[31] = readincr(cb_input);
    //     b[32+0] = readincr(cb_input);
    //     b[32+1] = readincr(cb_input);
    //     b[32+2] = readincr(cb_input);
    //     b[32+3] = readincr(cb_input);
    //     b[32+4] = readincr(cb_input);
    //     b[32+5] = readincr(cb_input);
    //     b[32+6] = readincr(cb_input);
    //     b[32+7] = readincr(cb_input);
    //     b[32+8] = readincr(cb_input);
    //     b[32+9] = readincr(cb_input);
    //     b[32+10] = readincr(cb_input);
    //     b[32+11] = readincr(cb_input);
    //     b[32+12] = readincr(cb_input);
    //     b[32+13] = readincr(cb_input);
    //     b[32+14] = readincr(cb_input);
    //     b[32+15] = readincr(cb_input);
    //     b[32+16] = readincr(cb_input);
    //     b[32+17] = readincr(cb_input);
    //     b[32+18] = readincr(cb_input);
    //     b[32+19] = readincr(cb_input);
    //     b[32+20] = readincr(cb_input);
    //     b[32+21] = readincr(cb_input);
    //     b[32+22] = readincr(cb_input);
    //     b[32+23] = readincr(cb_input);
    //     b[32+24] = readincr(cb_input);
    //     b[32+25] = readincr(cb_input);
    //     b[32+26] = readincr(cb_input);
    //     b[32+27] = readincr(cb_input);
    //     b[32+28] = readincr(cb_input);
    //     b[32+29] = readincr(cb_input);
    //     b[32+30] = readincr(cb_input);
    //     b[32+31] = readincr(cb_input);
    //     {
    //         uint16_t la  = a[0] & 0xFFFF;
    //         uint16_t lb = b[0] & 0xFFFF;
    //         uint16_t hb = (b[0] >> 16) & 0xFFFF;

    //         a[0] = (a[0] & 0xFFFF0000) | hb;
    //         b[0] = ((uint32_t)la << 16) | lb;
    //     }

    //     // --- b[1] 和 a[1] 交换 ---    

    //     {
    //          uint16_t la = a[1] & 0xFFFF;
    //          uint16_t lb = b[1] & 0xFFFF;
    //          uint16_t hb = (b[1] >> 16) & 0xFFFF;

    //         a[1] = (a[1] & 0xFFFF0000) | hb;
    //         b[1] = ((uint32_t)la << 16) | lb;
    //     }

    //     // --- 立刻输出 a[0], a[1] ---
    //     writeincr(cb_output, a[0]);
    //     writeincr(cb_output, a[1]);


    //     {
        
    //         uint16_t la = a[2] & 0xFFFF;
    //         uint16_t lb = b[2] & 0xFFFF;
    //         uint16_t hb = (b[2] >> 16) & 0xFFFF;

    //         a[2] = (a[2] & 0xFFFF0000) | hb;
    //         b[2] = ((uint32_t)la << 16) | lb;
        
    //     }

    //     {
    //         uint16_t la = a[3] & 0xFFFF;
    //         uint16_t lb = b[3] & 0xFFFF;
    //         uint16_t hb = (b[3] >> 16) & 0xFFFF;

    //         a[3] = (a[3] & 0xFFFF0000) | hb;
    //         b[3] = ((uint32_t)la << 16) | lb;
    //     }

    //     writeincr(cb_output, a[2]);
    //     writeincr(cb_output, a[3]);


    //     {
    //         uint16_t la = a[4] & 0xFFFF;
    //         uint16_t lb = b[4] & 0xFFFF;
    //         uint16_t hb = (b[4] >> 16) & 0xFFFF;

    //         a[4] = (a[4] & 0xFFFF0000) | hb;
    //         b[4] = ((uint32_t)la << 16) | lb;
    //     }


    //     {
    //         uint16_t la = a[5] & 0xFFFF;
    //         uint16_t lb = b[5] & 0xFFFF;
    //         uint16_t hb = (b[5] >> 16) & 0xFFFF;

    //         a[5] = (a[5] & 0xFFFF0000) | hb;
    //         b[5] = ((uint32_t)la << 16) | lb;
    //     }

    //     writeincr(cb_output, a[4]);
    //     writeincr(cb_output, a[5]);


    //     {
    //         uint16_t la = a[6] & 0xFFFF;
    //         uint16_t lb = b[6] & 0xFFFF;
    //         uint16_t hb = (b[6] >> 16) & 0xFFFF;

    //         a[6] = (a[6] & 0xFFFF0000) | hb;
    //         b[6] = ((uint32_t)la << 16) | lb;
    //     }


    //     {
    //         uint16_t la = a[7] & 0xFFFF;
    //         uint16_t lb = b[7] & 0xFFFF;
    //         uint16_t hb = (b[7] >> 16) & 0xFFFF;

    //         a[7] = (a[7] & 0xFFFF0000) | hb;
    //         b[7] = ((uint32_t)la << 16) | lb;
    //     }

    //     writeincr(cb_output, a[6]);
    //     writeincr(cb_output, a[7]);
        

    //     {
    //         uint16_t la = a[8] & 0xFFFF;
    //         uint16_t lb = b[8] & 0xFFFF;
    //         uint16_t hb = (b[8] >> 16) & 0xFFFF;

    //         a[8] = (a[8] & 0xFFFF0000) | hb;
    //         b[8] = ((uint32_t)la << 16) | lb;
    //     }


    //     {
    //         uint16_t la = a[9] & 0xFFFF;
    //         uint16_t lb = b[9] & 0xFFFF;
    //         uint16_t hb = (b[9] >> 16) & 0xFFFF;

    //         a[9] = (a[9] & 0xFFFF0000) | hb;
    //         b[9] = ((uint32_t)la << 16) | lb;
    //     }

    //     writeincr(cb_output, a[8]);
    //     writeincr(cb_output, a[9]);


    //     {
    //         uint16_t la = a[10] & 0xFFFF;
    //         uint16_t lb = b[10] & 0xFFFF;
    //         uint16_t hb = (b[10] >> 16) & 0xFFFF;

    //         a[10] = (a[10] & 0xFFFF0000) | hb;
    //         b[10] = ((uint32_t)la << 16) | lb;
    //     }


    //     {
    //         uint16_t la = a[11] & 0xFFFF;
    //         uint16_t lb = b[11] & 0xFFFF;
    //         uint16_t hb = (b[11] >> 16) & 0xFFFF;

    //         a[11] = (a[11] & 0xFFFF0000) | hb;
    //         b[11] = ((uint32_t)la << 16) | lb;
    //     }

    //     writeincr(cb_output, a[10]);
    //     writeincr(cb_output, a[11]);


    //     {
    //         uint16_t la = a[12] & 0xFFFF;
    //         uint16_t lb = b[12] & 0xFFFF;
    //         uint16_t hb = (b[12] >> 16) & 0xFFFF;

    //         a[12] = (a[12] & 0xFFFF0000) | hb;
    //         b[12] = ((uint32_t)la << 16) | lb;
    //     }


    //     {
    //         uint16_t la = a[13] & 0xFFFF;
    //         uint16_t lb = b[13] & 0xFFFF;
    //         uint16_t hb = (b[13] >> 16) & 0xFFFF;

    //         a[13] = (a[13] & 0xFFFF0000) | hb;
    //         b[13] = ((uint32_t)la << 16) | lb;
    //     }

    //     writeincr(cb_output, a[12]);
    //     writeincr(cb_output, a[13]);


    //     {
    //         uint16_t la = a[14] & 0xFFFF;
    //         uint16_t lb = b[14] & 0xFFFF;
    //         uint16_t hb = (b[14] >> 16) & 0xFFFF;

    //         a[14] = (a[14] & 0xFFFF0000) | hb;
    //         b[14] = ((uint32_t)la << 16) | lb;
    //     }


    //     {
    //         uint16_t la = a[15] & 0xFFFF;
    //         uint16_t lb = b[15] & 0xFFFF;
    //         uint16_t hb = (b[15] >> 16) & 0xFFFF;

    //         a[15] = (a[15] & 0xFFFF0000) | hb;
    //         b[15] = ((uint32_t)la << 16) | lb;
    //     }

    //     writeincr(cb_output, a[14]);
    //     writeincr(cb_output, a[15]);

    //     {
    //         uint16_t la = a[16] & 0xFFFF;
    //         uint16_t lb = b[16] & 0xFFFF;
    //         uint16_t hb = (b[16] >> 16) & 0xFFFF;
        
    //         a[16] = (a[16] & 0xFFFF0000) | hb;
    //         b[16] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     {
    //         uint16_t la = a[17] & 0xFFFF;
    //         uint16_t lb = b[17] & 0xFFFF;
    //         uint16_t hb = (b[17] >> 16) & 0xFFFF;
        
    //         a[17] = (a[17] & 0xFFFF0000) | hb;
    //         b[17] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     writeincr(cb_output, a[16]);
    //     writeincr(cb_output, a[17]);
        
    //     {
    //         uint16_t la = a[18] & 0xFFFF;
    //         uint16_t lb = b[18] & 0xFFFF;
    //         uint16_t hb = (b[18] >> 16) & 0xFFFF;
        
    //         a[18] = (a[18] & 0xFFFF0000) | hb;
    //         b[18] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     {
    //         uint16_t la = a[19] & 0xFFFF;
    //         uint16_t lb = b[19] & 0xFFFF;
    //         uint16_t hb = (b[19] >> 16) & 0xFFFF;
        
    //         a[19] = (a[19] & 0xFFFF0000) | hb;
    //         b[19] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     writeincr(cb_output, a[18]);
    //     writeincr(cb_output, a[19]);
        
    //     {
    //         uint16_t la = a[20] & 0xFFFF;
    //         uint16_t lb = b[20] & 0xFFFF;
    //         uint16_t hb = (b[20] >> 16) & 0xFFFF;
        
    //         a[20] = (a[20] & 0xFFFF0000) | hb;
    //         b[20] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     {
    //         uint16_t la = a[21] & 0xFFFF;
    //         uint16_t lb = b[21] & 0xFFFF;
    //         uint16_t hb = (b[21] >> 16) & 0xFFFF;
        
    //         a[21] = (a[21] & 0xFFFF0000) | hb;
    //         b[21] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     writeincr(cb_output, a[20]);
    //     writeincr(cb_output, a[21]);
        
    //     {
    //         uint16_t la = a[22] & 0xFFFF;
    //         uint16_t lb = b[22] & 0xFFFF;
    //         uint16_t hb = (b[22] >> 16) & 0xFFFF;
        
    //         a[22] = (a[22] & 0xFFFF0000) | hb;
    //         b[22] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     {
    //         uint16_t la = a[23] & 0xFFFF;
    //         uint16_t lb = b[23] & 0xFFFF;
    //         uint16_t hb = (b[23] >> 16) & 0xFFFF;
        
    //         a[23] = (a[23] & 0xFFFF0000) | hb;
    //         b[23] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     writeincr(cb_output, a[22]);
    //     writeincr(cb_output, a[23]);
        
    //     {
    //         uint16_t la = a[24] & 0xFFFF;
    //         uint16_t lb = b[24] & 0xFFFF;
    //         uint16_t hb = (b[24] >> 16) & 0xFFFF;
        
    //         a[24] = (a[24] & 0xFFFF0000) | hb;
    //         b[24] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     {
    //         uint16_t la = a[25] & 0xFFFF;
    //         uint16_t lb = b[25] & 0xFFFF;
    //         uint16_t hb = (b[25] >> 16) & 0xFFFF;
        
    //         a[25] = (a[25] & 0xFFFF0000) | hb;
    //         b[25] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     writeincr(cb_output, a[24]);
    //     writeincr(cb_output, a[25]);
        
    //     {
    //         uint16_t la = a[26] & 0xFFFF;
    //         uint16_t lb = b[26] & 0xFFFF;
    //         uint16_t hb = (b[26] >> 16) & 0xFFFF;
        
    //         a[26] = (a[26] & 0xFFFF0000) | hb;
    //         b[26] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     {
    //         uint16_t la = a[27] & 0xFFFF;
    //         uint16_t lb = b[27] & 0xFFFF;
    //         uint16_t hb = (b[27] >> 16) & 0xFFFF;
        
    //         a[27] = (a[27] & 0xFFFF0000) | hb;
    //         b[27] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     writeincr(cb_output, a[26]);
    //     writeincr(cb_output, a[27]);
        
    //     {
    //         uint16_t la = a[28] & 0xFFFF;
    //         uint16_t lb = b[28] & 0xFFFF;
    //         uint16_t hb = (b[28] >> 16) & 0xFFFF;
        
    //         a[28] = (a[28] & 0xFFFF0000) | hb;
    //         b[28] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     {
    //         uint16_t la = a[29] & 0xFFFF;
    //         uint16_t lb = b[29] & 0xFFFF;
    //         uint16_t hb = (b[29] >> 16) & 0xFFFF;
        
    //         a[29] = (a[29] & 0xFFFF0000) | hb;
    //         b[29] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     writeincr(cb_output, a[28]);
    //     writeincr(cb_output, a[29]);
        
    //     {
    //         uint16_t la = a[30] & 0xFFFF;
    //         uint16_t lb = b[30] & 0xFFFF;
    //         uint16_t hb = (b[30] >> 16) & 0xFFFF;
        
    //         a[30] = (a[30] & 0xFFFF0000) | hb;
    //         b[30] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     {
    //         uint16_t la = a[31] & 0xFFFF;
    //         uint16_t lb = b[31] & 0xFFFF;
    //         uint16_t hb = (b[31] >> 16) & 0xFFFF;
        
    //         a[31] = (a[31] & 0xFFFF0000) | hb;
    //         b[31] = ((uint32_t)la << 16) | lb;
    //     }
        
    //     writeincr(cb_output, a[30]);
    //     writeincr(cb_output, a[31]);
        
    //     {
    //     uint16_t la = a[32] & 0xFFFF;
    //     uint16_t lb = b[32] & 0xFFFF;
    //     uint16_t hb = (b[32] >> 16) & 0xFFFF;
    //     a[32] = (a[32] & 0xFFFF0000) | hb;
    //     b[32] = ((uint32_t)la << 16) | lb;

    //     la = a[33] & 0xFFFF;
    //     lb = b[33] & 0xFFFF;
    //     hb = (b[33] >> 16) & 0xFFFF;
    //     a[33] = (a[33] & 0xFFFF0000) | hb;
    //     b[33] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[32]);
    //     writeincr(cb_output, a[33]);
    // }
    // {
    //     uint16_t la = a[34] & 0xFFFF;
    //     uint16_t lb = b[34] & 0xFFFF;
    //     uint16_t hb = (b[34] >> 16) & 0xFFFF;
    //     a[34] = (a[34] & 0xFFFF0000) | hb;
    //     b[34] = ((uint32_t)la << 16) | lb;

    //     la = a[35] & 0xFFFF;
    //     lb = b[35] & 0xFFFF;
    //     hb = (b[35] >> 16) & 0xFFFF;
    //     a[35] = (a[35] & 0xFFFF0000) | hb;
    //     b[35] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[34]);
    //     writeincr(cb_output, a[35]);
    // }
    // {
    //     uint16_t la = a[36] & 0xFFFF;
    //     uint16_t lb = b[36] & 0xFFFF;
    //     uint16_t hb = (b[36] >> 16) & 0xFFFF;
    //     a[36] = (a[36] & 0xFFFF0000) | hb;
    //     b[36] = ((uint32_t)la << 16) | lb;

    //     la = a[37] & 0xFFFF;
    //     lb = b[37] & 0xFFFF;
    //     hb = (b[37] >> 16) & 0xFFFF;
    //     a[37] = (a[37] & 0xFFFF0000) | hb;
    //     b[37] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[36]);
    //     writeincr(cb_output, a[37]);
    // }
    // {
    //     uint16_t la = a[38] & 0xFFFF;
    //     uint16_t lb = b[38] & 0xFFFF;
    //     uint16_t hb = (b[38] >> 16) & 0xFFFF;
    //     a[38] = (a[38] & 0xFFFF0000) | hb;
    //     b[38] = ((uint32_t)la << 16) | lb;

    //     la = a[39] & 0xFFFF;
    //     lb = b[39] & 0xFFFF;
    //     hb = (b[39] >> 16) & 0xFFFF;
    //     a[39] = (a[39] & 0xFFFF0000) | hb;
    //     b[39] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[38]);
    //     writeincr(cb_output, a[39]);
    // }
    // {
    //     uint16_t la = a[40] & 0xFFFF;
    //     uint16_t lb = b[40] & 0xFFFF;
    //     uint16_t hb = (b[40] >> 16) & 0xFFFF;
    //     a[40] = (a[40] & 0xFFFF0000) | hb;
    //     b[40] = ((uint32_t)la << 16) | lb;

    //     la = a[41] & 0xFFFF;
    //     lb = b[41] & 0xFFFF;
    //     hb = (b[41] >> 16) & 0xFFFF;
    //     a[41] = (a[41] & 0xFFFF0000) | hb;
    //     b[41] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[40]);
    //     writeincr(cb_output, a[41]);
    // }
    // {
    //     uint16_t la = a[42] & 0xFFFF;
    //     uint16_t lb = b[42] & 0xFFFF;
    //     uint16_t hb = (b[42] >> 16) & 0xFFFF;
    //     a[42] = (a[42] & 0xFFFF0000) | hb;
    //     b[42] = ((uint32_t)la << 16) | lb;

    //     la = a[43] & 0xFFFF;
    //     lb = b[43] & 0xFFFF;
    //     hb = (b[43] >> 16) & 0xFFFF;
    //     a[43] = (a[43] & 0xFFFF0000) | hb;
    //     b[43] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[42]);
    //     writeincr(cb_output, a[43]);
    // }
    // {
    //     uint16_t la = a[44] & 0xFFFF;
    //     uint16_t lb = b[44] & 0xFFFF;
    //     uint16_t hb = (b[44] >> 16) & 0xFFFF;
    //     a[44] = (a[44] & 0xFFFF0000) | hb;
    //     b[44] = ((uint32_t)la << 16) | lb;

    //     la = a[45] & 0xFFFF;
    //     lb = b[45] & 0xFFFF;
    //     hb = (b[45] >> 16) & 0xFFFF;
    //     a[45] = (a[45] & 0xFFFF0000) | hb;
    //     b[45] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[44]);
    //     writeincr(cb_output, a[45]);
    // }
    // {
    //     uint16_t la = a[46] & 0xFFFF;
    //     uint16_t lb = b[46] & 0xFFFF;
    //     uint16_t hb = (b[46] >> 16) & 0xFFFF;
    //     a[46] = (a[46] & 0xFFFF0000) | hb;
    //     b[46] = ((uint32_t)la << 16) | lb;

    //     la = a[47] & 0xFFFF;
    //     lb = b[47] & 0xFFFF;
    //     hb = (b[47] >> 16) & 0xFFFF;
    //     a[47] = (a[47] & 0xFFFF0000) | hb;
    //     b[47] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[46]);
    //     writeincr(cb_output, a[47]);
    // }
    // {
    //     uint16_t la = a[48] & 0xFFFF;
    //     uint16_t lb = b[48] & 0xFFFF;
    //     uint16_t hb = (b[48] >> 16) & 0xFFFF;
    //     a[48] = (a[48] & 0xFFFF0000) | hb;
    //     b[48] = ((uint32_t)la << 16) | lb;

    //     la = a[49] & 0xFFFF;
    //     lb = b[49] & 0xFFFF;
    //     hb = (b[49] >> 16) & 0xFFFF;
    //     a[49] = (a[49] & 0xFFFF0000) | hb;
    //     b[49] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[48]);
    //     writeincr(cb_output, a[49]);
    // }
    // {
    //     uint16_t la = a[50] & 0xFFFF;
    //     uint16_t lb = b[50] & 0xFFFF;
    //     uint16_t hb = (b[50] >> 16) & 0xFFFF;
    //     a[50] = (a[50] & 0xFFFF0000) | hb;
    //     b[50] = ((uint32_t)la << 16) | lb;

    //     la = a[51] & 0xFFFF;
    //     lb = b[51] & 0xFFFF;
    //     hb = (b[51] >> 16) & 0xFFFF;
    //     a[51] = (a[51] & 0xFFFF0000) | hb;
    //     b[51] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[50]);
    //     writeincr(cb_output, a[51]);
    // }
    // {
    //     uint16_t la = a[52] & 0xFFFF;
    //     uint16_t lb = b[52] & 0xFFFF;
    //     uint16_t hb = (b[52] >> 16) & 0xFFFF;
    //     a[52] = (a[52] & 0xFFFF0000) | hb;
    //     b[52] = ((uint32_t)la << 16) | lb;

    //     la = a[53] & 0xFFFF;
    //     lb = b[53] & 0xFFFF;
    //     hb = (b[53] >> 16) & 0xFFFF;
    //     a[53] = (a[53] & 0xFFFF0000) | hb;
    //     b[53] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[52]);
    //     writeincr(cb_output, a[53]);
    // }
    // {
    //     uint16_t la = a[54] & 0xFFFF;
    //     uint16_t lb = b[54] & 0xFFFF;
    //     uint16_t hb = (b[54] >> 16) & 0xFFFF;
    //     a[54] = (a[54] & 0xFFFF0000) | hb;
    //     b[54] = ((uint32_t)la << 16) | lb;

    //     la = a[55] & 0xFFFF;
    //     lb = b[55] & 0xFFFF;
    //     hb = (b[55] >> 16) & 0xFFFF;
    //     a[55] = (a[55] & 0xFFFF0000) | hb;
    //     b[55] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[54]);
    //     writeincr(cb_output, a[55]);
    // }
    // {
    //     uint16_t la = a[56] & 0xFFFF;
    //     uint16_t lb = b[56] & 0xFFFF;
    //     uint16_t hb = (b[56] >> 16) & 0xFFFF;
    //     a[56] = (a[56] & 0xFFFF0000) | hb;
    //     b[56] = ((uint32_t)la << 16) | lb;

    //     la = a[57] & 0xFFFF;
    //     lb = b[57] & 0xFFFF;
    //     hb = (b[57] >> 16) & 0xFFFF;
    //     a[57] = (a[57] & 0xFFFF0000) | hb;
    //     b[57] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[56]);
    //     writeincr(cb_output, a[57]);
    // }
    // {
    //     uint16_t la = a[58] & 0xFFFF;
    //     uint16_t lb = b[58] & 0xFFFF;
    //     uint16_t hb = (b[58] >> 16) & 0xFFFF;
    //     a[58] = (a[58] & 0xFFFF0000) | hb;
    //     b[58] = ((uint32_t)la << 16) | lb;

    //     la = a[59] & 0xFFFF;
    //     lb = b[59] & 0xFFFF;
    //     hb = (b[59] >> 16) & 0xFFFF;
    //     a[59] = (a[59] & 0xFFFF0000) | hb;
    //     b[59] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[58]);
    //     writeincr(cb_output, a[59]);
    // }
    // {
    //     uint16_t la = a[60] & 0xFFFF;
    //     uint16_t lb = b[60] & 0xFFFF;
    //     uint16_t hb = (b[60] >> 16) & 0xFFFF;
    //     a[60] = (a[60] & 0xFFFF0000) | hb;
    //     b[60] = ((uint32_t)la << 16) | lb;

    //     la = a[61] & 0xFFFF;
    //     lb = b[61] & 0xFFFF;
    //     hb = (b[61] >> 16) & 0xFFFF;
    //     a[61] = (a[61] & 0xFFFF0000) | hb;
    //     b[61] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[60]);
    //     writeincr(cb_output, a[61]);
    // }
    // {
    //     uint16_t la = a[62] & 0xFFFF;
    //     uint16_t lb = b[62] & 0xFFFF;
    //     uint16_t hb = (b[62] >> 16) & 0xFFFF;
    //     a[62] = (a[62] & 0xFFFF0000) | hb;
    //     b[62] = ((uint32_t)la << 16) | lb;

    //     la = a[63] & 0xFFFF;
    //     lb = b[63] & 0xFFFF;
    //     hb = (b[63] >> 16) & 0xFFFF;
    //     a[63] = (a[63] & 0xFFFF0000) | hb;
    //     b[63] = ((uint32_t)la << 16) | lb;

    //     writeincr(cb_output, a[62]);
    //     writeincr(cb_output, a[63]);
    // }


    //     for(int j=0 ;j<64;j++)chess_prepare_for_pipelining
    //         chess_loop_range(32,) {
    //         writeincr(cb_output,b[j]);
    //     }


        //======================================//
        // SIMD technique, 8us
        //=====================================// 

            aie::vector<int32,32> temp_a1 = readincr_v<32>(cb_input);
            aie::vector<int32,32> temp_a2 = readincr_v<32>(cb_input);

            aie::vector<int32,32> temp_b1 = readincr_v<32>(cb_input);
            aie::vector<int32,32> temp_b2 = readincr_v<32>(cb_input);

            aie::vector<int16,64> a1 = as_v64int16(temp_a1); 
            aie::vector<int16,64> b1 = as_v64int16(temp_b1); 
            aie::vector<int16,64> a2 = as_v64int16(temp_a2); 
            aie::vector<int16,64> b2 = as_v64int16(temp_b2); 
             

            aie::vector<int16,32> a1_high = aie::filter_odd (a1);
            aie::vector<int16,32> a1_low =  aie::filter_even(a1);
            aie::vector<int16,32> b1_high = aie::filter_odd (b1);
            aie::vector<int16,32> b1_low =  aie::filter_even(b1);

            aie::vector<int16,32> a2_high = aie::filter_odd (a2,1);
            aie::vector<int16,32> a2_low =  aie::filter_even(a2,1);
            aie::vector<int16,32> b2_high = aie::filter_odd (b2,1);
            aie::vector<int16,32> b2_low =  aie::filter_even(b2,1);

            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r1 = aie::interleave_zip(a1_high,b1_high,1);
            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r2 = aie::interleave_zip(a1_low,b1_low,1);

            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r3 = aie::interleave_zip(a2_high,b2_high,1);
            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r4 = aie::interleave_zip(a2_low,b2_low,1);


            aie::vector<int32, 16> res1_first = as_v16int32(r1.first);
            aie::vector<int32, 16> res1_second = as_v16int32(r1.second);

            aie::vector<int32, 16> res2_first = as_v16int32(r2.first);
            aie::vector<int32, 16> res2_second = as_v16int32(r2.second);

            aie::vector<int32, 16> res3_first = as_v16int32(r3.first);
            aie::vector<int32, 16> res3_second = as_v16int32(r3.second);

            aie::vector<int32, 16> res4_first = as_v16int32(r4.first);
            aie::vector<int32, 16> res4_second = as_v16int32(r4.second);


            writeincr_v(cb_output,res1_first);
            writeincr_v(cb_output,res1_second);
            writeincr_v(cb_output,res2_first);
            writeincr_v(cb_output,res2_second);
            writeincr_v(cb_output,res3_first);
            writeincr_v(cb_output,res3_second);
            writeincr_v(cb_output,res4_first);
            writeincr_v(cb_output,res4_second);
    }       
   
}

