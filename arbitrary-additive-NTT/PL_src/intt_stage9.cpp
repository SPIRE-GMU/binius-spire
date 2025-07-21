// #include "binary_tower.h"  //to avoid recursive error during compilatin, I fixed the parameter for specific use
#include <ap_int.h>
#include <cstdint>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <stdint.h>
#include<iostream>
#include <sys/types.h>

#include "s9_conff1.h"
// #include "../common.h"
inline ap_uint<4> b_mul_alpha_u4(ap_uint<4> a) {
#pragma HLS INLINE
    ap_uint<4> res;
    res[0] = a[2];
    res[1] = a[3];
    res[2] = a[0] ^ a[3];
    res[3] = a[1] ^ a[2] ^ a[3];
    return res;
}

// Multiply by alpha for 8-bit input
inline ap_uint<8> b_mul_alpha_u8(ap_uint<8> a) {
#pragma HLS INLINE
    ap_uint<8> res;
    res[0] = a[4];
    res[1] = a[5];
    res[2] = a[6];
    res[3] = a[7];
    res[4] = a[0] ^ a[6];
    res[5] = a[1] ^ a[7];
    res[6] = a[2] ^ a[4] ^ a[7];
    res[7] = a[3] ^ a[5] ^ a[6] ^ a[7];
    return res;
}

// Multiply by alpha for 16-bit input using 8-bit alpha function
inline ap_uint<16> b_mul_alpha_u16(ap_uint<16> a) {
#pragma HLS INLINE
    ap_uint<8> a0 = a;
    ap_uint<8> a1 = a >> 8;
    return (ap_uint<16>)((a0 ^ b_mul_alpha_u8(a1)) | (a1 << 8));
}

// Multiply two 4-bit inputs in binary field
inline ap_uint<4> b_mul_u4(ap_uint<4> a, ap_uint<4> b) {
#pragma HLS INLINE
    ap_uint<4> res;
    res[0] = (a[0] & b[0]) ^ (a[1] & b[1]) ^ (a[2] & b[2]) ^ (a[3] & b[3]);
    res[1] = (a[1] & b[0]) ^ (a[0] & b[1]) ^ (a[1] & b[1]) ^ (a[3] & b[2]) ^ (a[2] & b[3]) ^ (a[3] & b[3]);
    res[2] = (a[2] & b[0]) ^ (a[3] & b[1]) ^ (a[0] & b[2]) ^ (a[3] & b[2]) ^ (a[1] & b[3]) ^ (a[2] & b[3]) ^ (a[3] & b[3]);
    res[3] = (a[3] & b[0]) ^ (a[2] & b[1]) ^ (a[3] & b[1]) ^ (a[1] & b[2]) ^ (a[2] & b[2]) ^ (a[3] & b[2]) ^
             (a[0] & b[3]) ^ (a[1] & b[3]) ^ (a[2] & b[3]);
    return res;
}

// Multiply two 8-bit inputs using 4-bit units and Karatsuba-like method
inline ap_uint<8> b_mul_u8(ap_uint<8> a, ap_uint<8> b) {
#pragma HLS INLINE
    ap_uint<4> a0 = a;
    ap_uint<4> a1 = a >> 4;
    ap_uint<4> b0 = b;
    ap_uint<4> b1 = b >> 4;
    ap_uint<4> z0 = b_mul_u4(a0, b0);
    ap_uint<4> z1 = b_mul_u4(a0 ^ a1, b0 ^ b1);
    ap_uint<4> z2 = b_mul_u4(a1, b1);
    ap_uint<4> low = z0 ^ z2;
    ap_uint<4> high = (z1 ^ z0 ^ z2) ^ b_mul_alpha_u4(z2);
    return (ap_uint<8>)((high, low));
}

// Multiply two 16-bit inputs using 8-bit units and Karatsuba-like method
inline ap_uint<16> b_mul_u16(ap_uint<16> a, ap_uint<16> b) {
#pragma HLS INLINE
    ap_uint<8> a0 = a;
    ap_uint<8> a1 = a >> 8;
    ap_uint<8> b0 = b;
    ap_uint<8> b1 = b >> 8;
    ap_uint<8> z0 = b_mul_u8(a0, b0);
    ap_uint<8> z1 = b_mul_u8(a0 ^ a1, b0 ^ b1);
    ap_uint<8> z2 = b_mul_u8(a1, b1);
    ap_uint<8> low = z0 ^ z2;
    ap_uint<8> high = (z1 ^ z0 ^ z2) ^ b_mul_alpha_u8(z2);
    return (ap_uint<16>)((high, low));
}

extern "C" {
void intt_stage9(hls::stream<ap_axis<32, 0, 0, 0> > &input, hls::stream<ap_axis<32, 0, 0, 0> > &output, int size) {
//#pragma HLS PIPELINE II=1
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS interface s_axilite port=return bundle=control
#pragma HLS interface s_axilite port=size bundle=control
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE axis port=input

// printf("PL kernel works");
uint16_t value_left, value_right;
uint16_t tempR,tempL;
uint16_t resL,resR;
uint32_t result;
for (int i=0; i<size; i++)
    {
          //std::cerr << "Waiting for a value" << "\n";
    
          ap_axis<32, 0, 0, 0> out_x;
          ap_axis<32, 0, 0, 0> in_x = input.read();
        //   printf("PL kernel read index%d is %d\n",i,in_x.data);
          
          value_right = in_x.data & 0xFFFF; //sample.right;
          value_left = (in_x.data >> 16) & 0xFFFF; // sample.left;

          tempR= b_mul_u16(value_right,W[i/512]);
          tempL =   b_mul_u16(value_left,W[i/512]^1);  //only 1024 coeff
    
          resL = tempR ^ tempL;
    
          resR = value_left ^ value_right;

            result = (((uint32_t)resL )<<16) |resR;
    
          out_x.data = result;
        
    
          //****out_x.keep_all();
          output.write(out_x);
    }
};

}
