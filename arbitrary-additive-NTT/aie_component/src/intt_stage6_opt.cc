#include <adf.h>
#include "adf/window/types.h"
#include "kernels.h"
#include "adf/intrinsics.h"
#include "adf/stream/types.h"
#include "aie_api/aie.hpp"
// #include "adf/x86sim/streamApi.h"
// #include "include.h"



//from now on we use for loop for generalizaton
void intt_stage6_opt(       input_stream_uint32 * cb_input,output_stream_uint32 * cb_output)
{
    uint32_t a[32],b[32];
    
    
    for(int i=0;i<2048/64;i++)
    chess_prepare_for_pipelining
    chess_loop_range(32,)    
    {   
        //=====================================================//
        //let compiler optimize the flatten instruction, 9 us  //
        //=====================================================//
    // //    a[0] = readi 
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

        

    //     for(int j=0 ;j<32;j++)chess_prepare_for_pipelining
    //         chess_loop_range(32,) {
    //         writeincr(cb_output,b[j]);
    //     }


        //======================================//
        // SIMD technique, 5us
        //=====================================// 

            aie::vector<int32,32> temp_a1 = readincr_v<32>(cb_input);
            aie::vector<int32,32> temp_b1 = readincr_v<32>(cb_input);
            aie::vector<int16,64> a1 = as_v64int16(temp_a1); 
            aie::vector<int16,64> b1 = as_v64int16(temp_b1); 
            aie::vector<int16,32> a1_high = aie::filter_odd (a1);
            aie::vector<int16,32> a1_low =  aie::filter_even(a1);
            aie::vector<int16,32> b1_high = aie::filter_odd (b1);
            aie::vector<int16,32> b1_low =  aie::filter_even(b1);


            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r1 = aie::interleave_zip(a1_high,b1_high,1);
            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r2 = aie::interleave_zip(a1_low,b1_low,1);

            aie::vector<int32, 16> res1_first = as_v16int32(r1.first);
            aie::vector<int32, 16> res1_second = as_v16int32(r1.second);

            aie::vector<int32, 16> res2_first = as_v16int32(r2.first);
            aie::vector<int32, 16> res2_second = as_v16int32(r2.second);

            writeincr_v(cb_output,res1_first);
            writeincr_v(cb_output,res1_second);
            writeincr_v(cb_output,res2_first);
            writeincr_v(cb_output,res2_second);
    }       
   
}

