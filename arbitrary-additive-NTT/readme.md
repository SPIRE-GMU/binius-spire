# Configuration 

Ubuntu 24.04

Vitis 2025.1

Power Design Management 2025.2 (PDM)

# Performance of proposed ANTT architecture in different lengths.

The following table lists the resource consumption, latency, and power consumption of the proposed ANTT architecture in different lengths. 

We provide the ANTT  with common lengths of 512, 1024 and 2048 as an reference. Generally, the accelerator for a longer ANTT task requires more hardware resource and AI engine (AIE), which also comes out to a higher dynamic power consumption. The key advantage of the proposed architecture is that, the it maintains the throughput at around 10 Gb/s in arbitrary length of ANTT. Moreover, it is a general-purpose architecture, meaning it is applicable to other situations involving different length, by adding or removing permutation unit (PU) and butterfly unit (BU) kernels.  



| Length       | LUT(900K)     | FF(1.8M)      | BRAM | AIE | Freq(MHz) | Throughput(Gb/s) | Latency(us) | static(W) | dynamic(W) | power(W) |
|--------------|---------------|---------------|------|-----|-----------|------------------|-------------|-----------|------------|----------|
| 4096         | 14271(1.59%)  | 18132(1.01%)  | 3    | 12  |314        | 10.039           | 6.549       | 4.406     | 3.779      | 8.185    |
| 2048         | 12446(1.38%)  | 15482(0.86%)  | 3    | 11  |294        | 9.346            | 3.506       | 4.406     | 3.609      | 8.015    |
| 1024         | 12027(1.34%)  | 14953(0.83%)  | 3    | 10  |302        | 9.664            | 1.718       | 4.406     | 3.439      | 7.845    |
| 512          | 11540(1.28%)  | 14489(0.81%)  | 3    | 9   |317        | 10.076           | 0.813       | 4.406     | 3.269      | 7.675    |


# How to rescale the architecture

## AIE component 

  First, for the AIE graph, remove/add extra AIE kernel, as well as in/out ports. Then, reconnect the streams to correct kernel. After that, place the FIFO and kernel to a proper location.

## Hostcomponent

  On the host side, initialize the PL kernels according to the targeted length. Open and close these kernels in proper time.

## System component

  Add AIE graph and PL kernels to the binary-container file, also set the interconnection accordingly in the hardware-link.cfg file.

# Lazy solution

I update a new folder "lazy reproduce", which provide a way more easier approach to reproduce the architecture. Here are the instructions:

    Open vitis 2025.1 IDE
    
    Flow -> new component -> HLS conponent -> import source file and testbench 

    Change the parameter in testcase for customized test.   

    Synthesize & Run & Implementation 

The updated file reproduce the proposed IANTT with the same throughput, and you can find detailed report of timing, resource utilization as well. If any one have interest, I will update the left ANTT part as well as its system configuration during the conference. 
