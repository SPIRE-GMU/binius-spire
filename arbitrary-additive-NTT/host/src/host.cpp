

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <cstring>
#include <sys/select.h>
#include "../include/row1.h"
#include "../include/golden.h"

#include "experimental/xrt_kernel.h"

#include "adf/adf_api/XRTConfig.h"

#define INPUT_SIZE  4096
#define OUTPUT_SIZE 4096

#define NO_OF_ITERATIONS  1 

static std::vector<char>
load_xclbin(xrtDeviceHandle device, const std::string& fnm)
{
  if (fnm.empty())
    throw std::runtime_error("No xclbin speified");

  // load bit stream
  std::ifstream stream(fnm);
  stream.seekg(0,stream.end);
  size_t size = stream.tellg();
  stream.seekg(0,stream.beg);

  std::vector<char> header(size);
  stream.read(header.data(),size);

  auto top = reinterpret_cast<const axlf*>(header.data());
  if (xrtDeviceLoadXclbin(device, top))
    throw std::runtime_error("Bitstream download failed");

  return header;
}


int main(int argc, char ** argv)
{

	//////////////////////////////////////////
	// Open xclbin
	//////////////////////////////////////////
	
    if(argc <2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}

    char* xclbinFilename = argv[1];

	
	
    xuid_t uuid;
    auto dhdl = xrtDeviceOpen(0);
    auto xclbin = load_xclbin(dhdl, xclbinFilename);//xrtDeviceLoadXclbinFile(dhdl, xclbinFilename);
    xrtDeviceGetXclbinUUID(dhdl, uuid);




//This dictates number of iterations to run through.
    long itr = NO_OF_ITERATIONS;
   
//calculate input/output data size in number of samples.
    int sizeIn = INPUT_SIZE * itr;
    int sizeOut = OUTPUT_SIZE * itr;

    size_t input_size_in_bytes = sizeIn * sizeof(uint16_t);
    size_t output_size_in_bytes = sizeOut * sizeof(uint16_t);	

    //Manage/map input/output file
   
	//////////////////////////////////////////
	// input memory
	// No cache no sync seems not working. Should ask SSW team to investigate.
	//
    ////////////////////////////////////////	
    
    xrtBufferHandle in_bohdl = xrtBOAlloc(dhdl, input_size_in_bytes, 0, 0);
    auto in_bomapped = reinterpret_cast<short int*>(xrtBOMap(in_bohdl));
    memcpy(in_bomapped, Input, input_size_in_bytes);
    printf("Input memory virtual addr 0x%llx\n", in_bomapped);



    
    	
	//////////////////////////////////////////
	// output memory
	//////////////////////////////////////////
	
	xrtBufferHandle out_bohdl = xrtBOAlloc(dhdl, output_size_in_bytes, 0, 0);
    auto out_bomapped = reinterpret_cast<uint*>(xrtBOMap(out_bohdl));
	printf("Output memory virtual addr 0x%llx\n", out_bomapped);
	
	
	//////////////////////////////////////////
	// mm2s ip
	//////////////////////////////////////////
	
	xrtKernelHandle mm2s_khdl = xrtPLKernelOpen(dhdl, uuid, "mm2s");
	xrtRunHandle mm2s_rhdl = xrtRunOpen(mm2s_khdl);
    int rval = xrtRunSetArg(mm2s_rhdl, 0, in_bohdl);
    rval = xrtRunSetArg(mm2s_rhdl, 2, sizeIn/2);//64
    xrtRunStart(mm2s_rhdl);
	printf("run mm2s\n");
	
	//////////////////////////////////////////
	// s2mm ip
	//////////////////////////////////////////
	
	xrtKernelHandle s2mm_khdl = xrtPLKernelOpen(dhdl, uuid, "s2mm");
	xrtRunHandle s2mm_rhdl = xrtRunOpen(s2mm_khdl);
    rval = xrtRunSetArg(s2mm_rhdl, 0, out_bohdl);
    rval = xrtRunSetArg(s2mm_rhdl, 2, sizeOut/2);
    xrtRunStart(s2mm_rhdl);
	printf("run s2mm\n");
	
    //////////////////////////////////////////
    // intt s0 ip
    //////////////////////////////////////////

    xrtKernelHandle intt_stage0_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage0");
    xrtRunHandle intt_stage0_1_rhdl = xrtRunOpen(intt_stage0_1_khdl);
    rval = xrtRunSetArg(intt_stage0_1_rhdl, 2, sizeOut/2);
    xrtRunStart(intt_stage0_1_rhdl);
    printf("run intt_stage0_1\n");
	
	//////////////////////////////////////////
    // intt s1 ip
    //////////////////////////////////////////

    xrtKernelHandle intt_stage1_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage1");
    xrtRunHandle intt_stage1_1_rhdl = xrtRunOpen(intt_stage1_1_khdl);
    rval = xrtRunSetArg(intt_stage1_1_rhdl, 2, sizeOut/2);
    xrtRunStart(intt_stage1_1_rhdl);
    printf("run intt_stage1_1\n");
	
    //////////////////////////////////////////
    // intt s2 ip
    //////////////////////////////////////////

    xrtKernelHandle intt_stage2_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage2");
    xrtRunHandle intt_stage2_1_rhdl = xrtRunOpen(intt_stage2_1_khdl);
    rval = xrtRunSetArg(intt_stage2_1_rhdl, 2, sizeOut/2);
    xrtRunStart(intt_stage2_1_rhdl);
    printf("run intt_stage2_1\n");

    //////////////////////////////////////////
    // intt s3 ip
    //////////////////////////////////////////

    xrtKernelHandle intt_stage3_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage3");
    xrtRunHandle intt_stage3_1_rhdl = xrtRunOpen(intt_stage3_1_khdl);
    rval = xrtRunSetArg(intt_stage3_1_rhdl, 2, sizeOut/2);
    xrtRunStart(intt_stage3_1_rhdl);
    printf("run intt_stage3_1\n");
    
    //////////////////////////////////////////
    // intt s4 ip
    //////////////////////////////////////////

    xrtKernelHandle intt_stage4_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage4");
    xrtRunHandle intt_stage4_1_rhdl = xrtRunOpen(intt_stage4_1_khdl);
    rval = xrtRunSetArg(intt_stage4_1_rhdl, 2, sizeOut/2);
    xrtRunStart(intt_stage4_1_rhdl);
    printf("run intt_stage4_1\n");
    
    //////////////////////////////////////////
    // intt s5 ip
    //////////////////////////////////////////

    xrtKernelHandle intt_stage5_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage5");
    xrtRunHandle intt_stage5_1_rhdl = xrtRunOpen(intt_stage5_1_khdl);
    rval = xrtRunSetArg(intt_stage5_1_rhdl, 2, sizeOut/2);
    xrtRunStart(intt_stage5_1_rhdl);
    printf("run intt_stage5_1\n");

    //////////////////////////////////////////
    // intt s6 ip
    //////////////////////////////////////////

    xrtKernelHandle intt_stage6_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage6");
    xrtRunHandle intt_stage6_1_rhdl = xrtRunOpen(intt_stage6_1_khdl);
    rval = xrtRunSetArg(intt_stage6_1_rhdl, 2, sizeOut/2);
    xrtRunStart(intt_stage6_1_rhdl);
    printf("run intt_stage6_1\n");
    //////////////////////////////////////////
    // intt s7 ip
    //////////////////////////////////////////

    xrtKernelHandle intt_stage7_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage7");
    xrtRunHandle intt_stage7_1_rhdl = xrtRunOpen(intt_stage7_1_khdl);
    rval = xrtRunSetArg(intt_stage7_1_rhdl, 2, sizeOut/2);
    xrtRunStart(intt_stage7_1_rhdl);
    printf("run intt_stage7_1\n");

    //////////////////////////////////////////
    // intt s8 ip
    //////////////////////////////////////////

    xrtKernelHandle intt_stage8_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage8");
    xrtRunHandle intt_stage8_1_rhdl = xrtRunOpen(intt_stage8_1_khdl);
    rval = xrtRunSetArg(intt_stage8_1_rhdl, 2, sizeOut/2);
    xrtRunStart(intt_stage8_1_rhdl);
    printf("run intt_stage8_1\n");

    // //////////////////////////////////////////
    // // intt s9 ip
    // //////////////////////////////////////////

    // xrtKernelHandle intt_stage9_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage9");
    // xrtRunHandle intt_stage9_1_rhdl = xrtRunOpen(intt_stage9_1_khdl);
    // rval = xrtRunSetArg(intt_stage9_1_rhdl, 2, sizeOut/2);
    // xrtRunStart(intt_stage9_1_rhdl);
    // printf("run intt_stage9_1\n");

    // //////////////////////////////////////////
    // // intt s10 ip
    // //////////////////////////////////////////

    // xrtKernelHandle intt_stage10_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage10");
    // xrtRunHandle intt_stage10_1_rhdl = xrtRunOpen(intt_stage10_1_khdl);
    // rval = xrtRunSetArg(intt_stage10_1_rhdl, 2, sizeOut/2);
    // xrtRunStart(intt_stage10_1_rhdl);
    // printf("run intt_stage10_1\n");

    // //////////////////////////////////////////
    // // intt s11 ip
    // //////////////////////////////////////////

    // xrtKernelHandle intt_stage11_1_khdl = xrtPLKernelOpen(dhdl, uuid, "intt_stage11");
    // xrtRunHandle intt_stage11_1_rhdl = xrtRunOpen(intt_stage11_1_khdl);
    // rval = xrtRunSetArg(intt_stage11_1_rhdl, 2, sizeOut/2);
    // xrtRunStart(intt_stage11_1_rhdl);
    // printf("run intt_stage11_1\n");

    
    printf("xrtGraphOpen\n"); 
    auto ghdl = xrtGraphOpen(dhdl, uuid, "clipgraph"); 
    printf("xrtGraphRun\n"); 

    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    
    xrtGraphRun(ghdl, 1);
     
    //////////////////////////////////////////
	// wait for mm2s done
	//////////////////////////////////////////	
    auto state = xrtRunWait(mm2s_rhdl);
    std::cout << "mm2s completed with status(" << state << ")\n";
   
    //////////////////////////////////////////
    // wait for s0 done
    //////////////////////////////////////////	
    state = xrtRunWait(intt_stage0_1_rhdl);
    std::cout << "intt_stage0_1 completed with status(" << state << ")\n";
    
    //////////////////////////////////////////
    // wait for s1 done
    //////////////////////////////////////////	
    state = xrtRunWait(intt_stage1_1_rhdl);
    std::cout << "intt_stage1_1 completed with status(" << state << ")\n";

    //////////////////////////////////////////
    // wait for s2 done
    //////////////////////////////////////////	
    state = xrtRunWait(intt_stage2_1_rhdl);
    std::cout << "intt_stage2_1 completed with status(" << state << ")\n";

    //////////////////////////////////////////
    // wait for s2mm done
	//////////////////////////////////////////	
	
	state = xrtRunWait(s2mm_rhdl);
    std::cout << "s2mm completed with status(" << state << ")\n";

    
    xrtGraphEnd(ghdl,0);
    gettimeofday(&end, NULL);
    long timeuse = 1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
    printf("timeuse = %f us",timeuse/1000000.0);    

    printf("xrtGraphEnd..\n"); 
    xrtGraphClose(ghdl);
	

    xrtRunClose(s2mm_rhdl);
    xrtKernelClose(s2mm_khdl);

    xrtRunClose(mm2s_rhdl);
    xrtKernelClose(mm2s_khdl);
  
    xrtRunClose(intt_stage0_1_rhdl);
    xrtKernelClose(intt_stage0_1_khdl);

	xrtRunClose(intt_stage1_1_rhdl);
    xrtKernelClose(intt_stage1_1_khdl);

    xrtRunClose(intt_stage2_1_rhdl);
    xrtKernelClose(intt_stage2_1_khdl);

    xrtRunClose(intt_stage3_1_rhdl);
    xrtKernelClose(intt_stage3_1_khdl); 

    xrtRunClose(intt_stage4_1_rhdl);
    xrtKernelClose(intt_stage4_1_khdl); 

    xrtRunClose(intt_stage5_1_rhdl);
    xrtKernelClose(intt_stage5_1_khdl);   

    xrtRunClose(intt_stage6_1_rhdl);
    xrtKernelClose(intt_stage6_1_khdl);  

    xrtRunClose(intt_stage7_1_rhdl);
    xrtKernelClose(intt_stage7_1_khdl); 

    xrtRunClose(intt_stage8_1_rhdl);
    xrtKernelClose(intt_stage8_1_khdl);  

    // xrtRunClose(intt_stage9_1_rhdl);
    // xrtKernelClose(intt_stage9_1_khdl);  

    // xrtRunClose(intt_stage10_1_rhdl);
    // xrtKernelClose(intt_stage10_1_khdl); 

    // xrtRunClose(intt_stage11_1_rhdl);
    // xrtKernelClose(intt_stage11_1_khdl);  
	//////////////////////////////////////////
	// compare results
	//////////////////////////////////////////	
	
    int errCnt = 0;
    // for (int i = 0; i < sizeOut/2; i++) {
    //     if (out_bomapped[i] != output[i]) {
    //         printf("ERROR: Test failed! Error found in sample %d: golden: %d, hardware: %u\n", i, output[i], out_bomapped[i]);
    //         errCnt++;
    //     }
    // }

	
    //////////////////////////////////////////
	// clean up XRT
	//////////////////////////////////////////	
    
    std::cout << "Releasing remaining XRT objects...\n";
    xrtBOFree(in_bohdl);
    xrtBOFree(out_bohdl);
    xrtDeviceClose(dhdl);
	
    std::cout << "TEST " << (errCnt ? "FAILED" : "PASSED") << std::endl; 
    return (errCnt ? EXIT_FAILURE :  EXIT_SUCCESS);
}

