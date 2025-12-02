

#include <adf.h>
#include "adf/window/types.h"
#include "kernels.h"
#include "adf/intrinsics.h"
#include "adf/stream/types.h"
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/adf/stream.hpp"

#include "stdio.h"
// #include "adf/x86sim/streamApi.h"
// #include "include.h"




//from now on we use for loop for generalizaton
void intt_stage8_opt(       input_stream_int32 * cb_input,output_stream_int32 * cb_output)
{
    // uint32_t a[128],b[128];
    // v32int32 temp_a1,temp_a2,temp_a3,temp_a4;
    // v32int32 temp_b1,temp_b2,temp_b3,temp_b4;

    // v64int16 a1,a2,a3,a4;
    // v64int16 b1,b2,b3,b4;

    // v32int32 res_a1,res_a2,res_a3,res_a4; 
    // v32int32 res_b1,res_b2,res_b3,res_b4; 
    for(int i=0;i<2048/256;i++)
    chess_prepare_for_pipelining
    chess_loop_range(32,)  
    {
            //utiliza the SIMD, AIE will achieve a higher level throughput, whose latency is 5 us, which is better than instruction-level pipeine
            aie::vector<int32,32> temp_a1 = readincr_v<32>(cb_input);
            aie::vector<int32,32> temp_a2 = readincr_v<32>(cb_input);
            aie::vector<int32,32> temp_a3 = readincr_v<32>(cb_input);
            aie::vector<int32,32> temp_a4 = readincr_v<32>(cb_input);

            aie::vector<int32,32> temp_b1 = readincr_v<32>(cb_input);
            aie::vector<int32,32> temp_b2 = readincr_v<32>(cb_input);
            aie::vector<int32,32> temp_b3 = readincr_v<32>(cb_input);
            aie::vector<int32,32> temp_b4 = readincr_v<32>(cb_input);

            aie::vector<int16,64> a1 = as_v64int16(temp_a1); 
            aie::vector<int16,64> b1 = as_v64int16(temp_b1); 
            aie::vector<int16,64> a2 = as_v64int16(temp_a2); 
            aie::vector<int16,64> b2 = as_v64int16(temp_b2); 
            aie::vector<int16,64> a3 = as_v64int16(temp_a3); 
            aie::vector<int16,64> b3 = as_v64int16(temp_b3); 
            aie::vector<int16,64> a4 = as_v64int16(temp_a4); 
            aie::vector<int16,64> b4 = as_v64int16(temp_b4); 

            aie::vector<int16,32> a1_high = aie::filter_odd (a1);
            aie::vector<int16,32> a1_low =  aie::filter_even(a1);
            aie::vector<int16,32> b1_high = aie::filter_odd (b1);
            aie::vector<int16,32> b1_low =  aie::filter_even(b1);

            aie::vector<int16,32> a2_high = aie::filter_odd (a2,1);
            aie::vector<int16,32> a2_low =  aie::filter_even(a2,1);
            aie::vector<int16,32> b2_high = aie::filter_odd (b2,1);
            aie::vector<int16,32> b2_low =  aie::filter_even(b2,1);

            aie::vector<int16,32> a3_high = aie::filter_odd (a3,1);
            aie::vector<int16,32> a3_low =  aie::filter_even(a3,1);
            aie::vector<int16,32> b3_high = aie::filter_odd (b3,1);
            aie::vector<int16,32> b3_low =  aie::filter_even(b3,1);

            aie::vector<int16,32> a4_high = aie::filter_odd (a4,1);
            aie::vector<int16,32> a4_low =  aie::filter_even(a4,1);
            aie::vector<int16,32> b4_high = aie::filter_odd (b4,1);
            aie::vector<int16,32> b4_low =  aie::filter_even(b4,1);


            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r1 = aie::interleave_zip(b1_high,a1_high,1);
            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r2 = aie::interleave_zip(b1_low,a1_low,1);

            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r3 = aie::interleave_zip(b2_high,a2_high,1);
            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r4 = aie::interleave_zip(b2_low,a2_low,1);

            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r5 = aie::interleave_zip(b3_high,a3_high,1);
            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r6 = aie::interleave_zip(b3_low,a3_low,1);

            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r7 = aie::interleave_zip(b4_high,a4_high,1);
            std::pair<aie::vector<int16,32>,aie::vector<int16,32>> r8 = aie::interleave_zip(b4_low,a4_low,1);

            aie::vector<int32, 16> res1_first = as_v16int32(r1.first);
            aie::vector<int32, 16> res1_second = as_v16int32(r1.second);

            aie::vector<int32, 16> res2_first = as_v16int32(r2.first);
            aie::vector<int32, 16> res2_second = as_v16int32(r2.second);

            aie::vector<int32, 16> res3_first = as_v16int32(r3.first);
            aie::vector<int32, 16> res3_second = as_v16int32(r3.second);

            aie::vector<int32, 16> res4_first = as_v16int32(r4.first);
            aie::vector<int32, 16> res4_second = as_v16int32(r4.second);

            aie::vector<int32, 16> res5_first = as_v16int32(r5.first);
            aie::vector<int32, 16> res5_second = as_v16int32(r5.second);

            aie::vector<int32, 16> res6_first = as_v16int32(r6.first);
            aie::vector<int32, 16> res6_second = as_v16int32(r6.second);

            aie::vector<int32, 16> res7_first = as_v16int32(r7.first);
            aie::vector<int32, 16> res7_second = as_v16int32(r7.second);

            aie::vector<int32, 16> res8_first = as_v16int32(r8.first);
            aie::vector<int32, 16> res8_second = as_v16int32(r8.second);

            writeincr_v(cb_output,res1_first);
            writeincr_v(cb_output,res1_second);
            writeincr_v(cb_output,res2_first);
            writeincr_v(cb_output,res2_second);
            writeincr_v(cb_output,res3_first);
            writeincr_v(cb_output,res3_second);
            writeincr_v(cb_output,res4_first);
            writeincr_v(cb_output,res4_second);
            writeincr_v(cb_output,res5_first);
            writeincr_v(cb_output,res5_second);
            writeincr_v(cb_output,res6_first);
            writeincr_v(cb_output,res6_second);
            writeincr_v(cb_output,res7_first);
            writeincr_v(cb_output,res7_second);
            writeincr_v(cb_output,res8_first);
            writeincr_v(cb_output,res8_second);

    //==============================================================================================================
    //  it takes 14us 
    //  we cannot always unroll all the instructions as stage 7 does, because it it limited to 16KB program memory.
    //   
    //  as a result, we should try to use SIMD for higher  throughput
    //=============================================================================================================
        // for(int j=0 ;j<128;j++)chess_prepare_for_pipelining
        //     chess_loop_range(32,) {
        //     // v8uint16
        //     a[j] = readincr(cb_input);
        // }
        // for(int k=0 ;k<128;k++)chess_prepare_for_pipelining
        //     chess_loop_range(32,) {
        //     b[k] = readincr(cb_input);
        // }
        
        // for(int u=0;u<128;u+=8)chess_prepare_for_pipelining
        //     chess_loop_range(32,){
        //     // exchange(&a[u],&b[u]);  
            
            
        //     {
        //     uint16_t la  = a[u+0] & 0xFFFF;
        //     uint16_t lb = b[u+0] & 0xFFFF;
        //     uint16_t hb = (b[u+0] >> 16) & 0xFFFF;

        //     a[u+0] = (a[u+0] & 0xFFFF0000) | hb;
        //     b[u+0] = ((uint32_t)la << 16) | lb;
        //     }

        // // --- b[1] 和 a[1] 交换 ---    

        // {
        //      uint16_t la = a[u+1] & 0xFFFF;
        //      uint16_t lb = b[u+1] & 0xFFFF;
        //      uint16_t hb = (b[u+1] >> 16) & 0xFFFF;

        //     a[u+1] = (a[u+1] & 0xFFFF0000) | hb;
        //     b[u+1] = ((uint32_t)la << 16) | lb;
        // }

        // // --- 立刻输出 a[0], a[1] ---
        // // writeincr(cb_output, a[0]);
        // // writeincr(cb_output, a[1]);


        // {
        
        //     uint16_t la = a[u+2] & 0xFFFF;
        //     uint16_t lb = b[u+2] & 0xFFFF;
        //     uint16_t hb = (b[u+2] >> 16) & 0xFFFF;

        //     a[u+2] = (a[u+2] & 0xFFFF0000) | hb;
        //     b[u+2] = ((uint32_t)la << 16) | lb;
        
        // }

        // {
        //     uint16_t la = a[u+3] & 0xFFFF;
        //     uint16_t lb = b[u+3] & 0xFFFF;
        //     uint16_t hb = (b[u+3] >> 16) & 0xFFFF;

        //     a[u+3] = (a[u+3] & 0xFFFF0000) | hb;
        //     b[u+3] = ((uint32_t)la << 16) | lb;
        // }

        // // writeincr(cb_output, a[2]);
        // // writeincr(cb_output, a[3]);


        // {
        //     uint16_t la = a[u+4] & 0xFFFF;
        //     uint16_t lb = b[u+4] & 0xFFFF;
        //     uint16_t hb = (b[u+4] >> 16) & 0xFFFF;

        //     a[u+4] = (a[u+4] & 0xFFFF0000) | hb;
        //     b[u+4] = ((uint32_t)la << 16) | lb;
        // }


        // {
        //     uint16_t la = a[u+5] & 0xFFFF;
        //     uint16_t lb = b[u+5] & 0xFFFF;
        //     uint16_t hb = (b[u+5] >> 16) & 0xFFFF;

        //     a[u+5] = (a[u+5] & 0xFFFF0000) | hb;
        //     b[u+5] = ((uint32_t)la << 16) | lb;
        // }

        // // writeincr(cb_output, a[4]);
        // // writeincr(cb_output, a[5]);


        // {
        //     uint16_t la = a[6] & 0xFFFF;
        //     uint16_t lb = b[6] & 0xFFFF;
        //     uint16_t hb = (b[6] >> 16) & 0xFFFF;

        //     a[u+6] = (a[u+6] & 0xFFFF0000) | hb;
        //     b[u+6] = ((uint32_t)la << 16) | lb;
        // }


        // {
        //     uint16_t la = a[u+7] & 0xFFFF;
        //     uint16_t lb = b[u+7] & 0xFFFF;
        //     uint16_t hb = (b[u+7] >> 16) & 0xFFFF;

        //     a[u+7] = (a[u+7] & 0xFFFF0000) | hb;
        //     b[u+7] = ((uint32_t)la << 16) | lb;
        // }

        // // writeincr(cb_output, a[6]);
        // // writeincr(cb_output, a[7]);

        //     writeincr(cb_output,a[u]);
        //     writeincr(cb_output,a[u+1]);
        //     writeincr(cb_output,a[u+2]);
        //     writeincr(cb_output,a[u+3]);
        //     writeincr(cb_output,a[u+4]);
        //     writeincr(cb_output,a[u+5]);
        //     writeincr(cb_output,a[u+6]);
        //     writeincr(cb_output,a[u+7]);

        // }
   
        
        // // for(int j=0 ;j<128;j++)chess_prepare_for_pipelining
        // //     chess_loop_range(32,) {
        // //     writeincr(cb_output,a[j]);
        // // }
        // for(int j=0 ;j<128;j++)chess_prepare_for_pipelining
        //     chess_loop_range(32,) {
        //     writeincr(cb_output,b[j]);
        // }



    }       
   



        
   
}

