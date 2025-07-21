#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
// #include "include.h"
#include "adf/new_frontend/adf.h"
#include "kernels.h"


using namespace adf;

class clipped : public graph {  

  private:
    kernel AIE_intt_stage_0,AIE_intt_stage_1,AIE_intt_stage_2,AIE_intt_stage_3,AIE_intt_stage_4,AIE_intt_stage_5,AIE_intt_stage_6,AIE_intt_stage_7,AIE_intt_stage_8,AIE_intt_stage_9,AIE_intt_stage_10,AIE_intt_stage_11;
    kernel cache;
   
  public:
 
    adf::input_plio in0,PL_intt_stage_0_out,PL_intt_stage_1_out,PL_intt_stage_2_out,PL_intt_stage_3_out,PL_intt_stage_4_out,PL_intt_stage_5_out,PL_intt_stage_6_out,PL_intt_stage_7_out,PL_intt_stage_8_out,PL_intt_stage_9_out,PL_intt_stage_10_out,PL_intt_stage_11_out;
	adf::output_plio  out0,PL_intt_stage_0_in,PL_intt_stage_1_in,PL_intt_stage_2_in,PL_intt_stage_3_in,PL_intt_stage_4_in,PL_intt_stage_5_in,PL_intt_stage_6_in,PL_intt_stage_7_in,PL_intt_stage_8_in,PL_intt_stage_9_in,PL_intt_stage_10_in,PL_intt_stage_11_in;
    clipped() {
	in0 = adf::input_plio::create("DataIn1", adf::plio_32_bits,"data/input.txt");
    out0 = adf::output_plio::create("DataOut1",adf::plio_32_bits, "data/output.txt");
	PL_intt_stage_0_out =  adf::input_plio::create("Data_clip_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_0_in  =  adf::output_plio::create("Data_clip_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_1_out =  adf::input_plio::create("Data_stage1_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_1_in  =  adf::output_plio::create("Data_stage1_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_2_out =  adf::input_plio::create("Data_stage2_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_2_in  =  adf::output_plio::create("Data_stage2_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_3_out =  adf::input_plio::create("Data_stage3_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_3_in  =  adf::output_plio::create("Data_stage3_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_4_out =  adf::input_plio::create("Data_stage4_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_4_in  =  adf::output_plio::create("Data_stage4_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_5_out =  adf::input_plio::create("Data_stage5_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_5_in  =  adf::output_plio::create("Data_stage5_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_6_out =  adf::input_plio::create("Data_stage6_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_6_in  =  adf::output_plio::create("Data_stage6_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_7_out =  adf::input_plio::create("Data_stage7_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_7_in  =  adf::output_plio::create("Data_stage7_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_8_out =  adf::input_plio::create("Data_stage8_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_8_in  =  adf::output_plio::create("Data_stage8_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_9_out =  adf::input_plio::create("Data_stage9_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_9_in  =  adf::output_plio::create("Data_stage9_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_10_out =  adf::input_plio::create("Data_stage10_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_10_in  =  adf::output_plio::create("Data_stage10_in0", adf::plio_32_bits,"data/output1.txt");
    PL_intt_stage_11_out =  adf::input_plio::create("Data_stage11_out0",adf::plio_32_bits, "data/input1.txt");
	PL_intt_stage_11_in  =  adf::output_plio::create("Data_stage11_in0", adf::plio_32_bits,"data/output1.txt");


    AIE_intt_stage_0 = kernel::create(intt_stage0);
    AIE_intt_stage_1 = kernel::create(intt_stage1);
    AIE_intt_stage_2 = kernel::create(intt_stage2);
    AIE_intt_stage_3 = kernel::create(intt_stage3);
    AIE_intt_stage_4 = kernel::create(intt_stage4);
    AIE_intt_stage_5 = kernel::create(intt_stage5);
    AIE_intt_stage_6 = kernel::create(intt_stage6);
    AIE_intt_stage_7 = kernel::create(intt_stage7);
    AIE_intt_stage_8 = kernel::create(intt_stage8);
    AIE_intt_stage_9 = kernel::create(intt_stage9);
    AIE_intt_stage_10 = kernel::create(intt_stage10);
    AIE_intt_stage_11 = kernel::create(intt_stage11);

    cache     = kernel::create(rearrange_out);

      

        connect< stream > net0(in0.out[0], AIE_intt_stage_0.in[0]);
        connect< stream > net1(AIE_intt_stage_0.out[0], PL_intt_stage_0_in.in[0]);   
	    
        connect< stream > net2 (PL_intt_stage_0_out.out[0], AIE_intt_stage_1.in[0]); //easy to extend
        connect< stream > net3(AIE_intt_stage_1.out[0], PL_intt_stage_1_in.in[0]);
        
        connect< stream > net4 (PL_intt_stage_1_out.out[0], AIE_intt_stage_2.in[0]);
        connect< stream > net5(AIE_intt_stage_2.out[0], PL_intt_stage_2_in.in[0]);
        
        connect< stream > net6 (PL_intt_stage_2_out.out[0], AIE_intt_stage_3.in[0]);
        connect< stream > net7(AIE_intt_stage_3.out[0], PL_intt_stage_3_in.in[0]);

        connect< stream > net8 (PL_intt_stage_3_out.out[0], AIE_intt_stage_4.in[0]);
        connect< stream > net9(AIE_intt_stage_4.out[0], PL_intt_stage_4_in.in[0]);

        connect< stream > net10 (PL_intt_stage_4_out.out[0], AIE_intt_stage_5.in[0]);
        connect< stream > net11(AIE_intt_stage_5.out[0], PL_intt_stage_5_in.in[0]);
        
        connect< stream > net12 (PL_intt_stage_5_out.out[0], AIE_intt_stage_6.in[0]);
        connect< stream > net13(AIE_intt_stage_6.out[0], PL_intt_stage_6_in.in[0]);
        
        connect< stream > net14 (PL_intt_stage_6_out.out[0], AIE_intt_stage_7.in[0]);
        connect< stream > net15(AIE_intt_stage_7.out[0], PL_intt_stage_7_in.in[0]);

        connect< stream > net16 (PL_intt_stage_7_out.out[0], AIE_intt_stage_8.in[0]);
        connect< stream > net17(AIE_intt_stage_8.out[0], PL_intt_stage_8_in.in[0]);

        connect< stream > net18 (PL_intt_stage_8_out.out[0], AIE_intt_stage_9.in[0]);
        connect< stream > net19(AIE_intt_stage_9.out[0], PL_intt_stage_9_in.in[0]);

        connect< stream > net20 (PL_intt_stage_9_out.out[0], AIE_intt_stage_10.in[0]);
        connect< stream > net21(AIE_intt_stage_10.out[0], PL_intt_stage_10_in.in[0]);
        
        connect< stream > net22 (PL_intt_stage_10_out.out[0], AIE_intt_stage_11.in[0]);
        connect< stream > net23(AIE_intt_stage_11.out[0], PL_intt_stage_11_in.in[0]);

        
        connect< stream > net100 (PL_intt_stage_11_out.out[0],cache.in[0]);
        connect< stream > net101 (cache.out[0], out0.in[0]);
       
        fifo_depth(net0)=512;
        fifo_depth(net1)=512;
        fifo_depth(net2)=512;
        fifo_depth(net3)=512;
        fifo_depth(net4)=512;
        fifo_depth(net5)=512;
        fifo_depth(net6)=512;
        fifo_depth(net7)=512;
        fifo_depth(net8)=512;
        fifo_depth(net9)=512;
        fifo_depth(net10)=512;
        fifo_depth(net11)=512;
        fifo_depth(net12)=512;
        fifo_depth(net13)=512;
        fifo_depth(net14)=512;
        fifo_depth(net15)=512;
        fifo_depth(net16)=512;
        fifo_depth(net17)=512;
        fifo_depth(net18)=512;
        fifo_depth(net19)=512;
        fifo_depth(net20)=512;
        fifo_depth(net21)=512;
        fifo_depth(net22)=512;
        fifo_depth(net23)=512;
        
        fifo_depth(net100)=512;
        fifo_depth(net101)=512;

        source(AIE_intt_stage_0) = "intt_stage0.cc";
        source(AIE_intt_stage_1) = "intt_stage1.cc";
        source(AIE_intt_stage_2) = "intt_stage2.cc";
        source(AIE_intt_stage_3) = "intt_stage3.cc";
        source(AIE_intt_stage_4) = "intt_stage4.cc";
        source(AIE_intt_stage_5) = "intt_stage5.cc";
        source(AIE_intt_stage_6) = "intt_stage6.cc";
        source(AIE_intt_stage_7) = "intt_stage7.cc";
        source(AIE_intt_stage_8) = "intt_stage8.cc";
        source(AIE_intt_stage_9) = "intt_stage9.cc";
        source(AIE_intt_stage_10) = "intt_stage10.cc";
        source(AIE_intt_stage_11) = "intt_stage11.cc";

        source(cache)    = "rearrange_out.cc";
       
        adf::location<kernel>(AIE_intt_stage_0)= tile(6,1);
        adf::location<kernel>(AIE_intt_stage_1)= tile(7,1);
        adf::location<kernel>(AIE_intt_stage_2)= tile(8,1);
        adf::location<kernel>(AIE_intt_stage_3)= tile(9,1);
        adf::location<kernel>(AIE_intt_stage_4)= tile(10,1);
        adf::location<kernel>(AIE_intt_stage_5)= tile(11,1);
        adf::location<kernel>(AIE_intt_stage_6)= tile(12,1);
        adf::location<kernel>(AIE_intt_stage_7)= tile(13,1);
        adf::location<kernel>(AIE_intt_stage_8)= tile(14,1);
        adf::location<kernel>(AIE_intt_stage_9)= tile(15,1);
        adf::location<kernel>(AIE_intt_stage_10)= tile(16,1);
        adf::location<kernel>(AIE_intt_stage_11)= tile(17,1);

        adf::location<kernel>(cache)= tile(18,1); 
        
        runtime<ratio>(AIE_intt_stage_0) = 0.9;
        runtime<ratio>(AIE_intt_stage_1) = 0.9;
        runtime<ratio>(AIE_intt_stage_2) = 0.9;
        runtime<ratio>(AIE_intt_stage_3) = 0.9;
        runtime<ratio>(AIE_intt_stage_4) = 0.9;
        runtime<ratio>(AIE_intt_stage_5) = 0.9;
        runtime<ratio>(AIE_intt_stage_6) = 0.9;
        runtime<ratio>(AIE_intt_stage_7) = 0.9;
        runtime<ratio>(AIE_intt_stage_8) = 0.9;
        runtime<ratio>(AIE_intt_stage_9) = 0.9;
        runtime<ratio>(AIE_intt_stage_10) = 0.9;
        runtime<ratio>(AIE_intt_stage_11) = 0.9;

        runtime<ratio>(cache) = 0.9;

	 
    };
};

#endif /* __GRAPH_H__ */
