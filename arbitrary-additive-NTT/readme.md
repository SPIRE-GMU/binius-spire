# Configuration 

Ubuntu 24.04

Vitis 2025.2

Power Design Management 2025.2 (PDM)

# Performance of proposed ANTT architecture in different lengths.

The following table lists the resource consumption, latency, and power consumption of the proposed ANTT architecture in different lengths. 

We provide the ANTT  with common lengths of 512, 1024 and 2048 as an reference. Generally, the accelerator for a longer ANTT task requires more hardware resource and AI engine (AIE), which also comes out to a higher dynamic power consumption. The key advantage of the proposed architecture is that, the it maintains the throughput at around 10 Gb/s in arbitrary length of ANTT. Moreover, it is a general-purpose architecture, meaning it is applicable to other situations involving different length, by adding or removing permutation unit (PU) and butterfly unit (BU) kernels.  



| Length       | LUT(80K)      | FF(1.8M)      | BRAM | AIE | Freq(MHz) | Throughput(Gb/s) | Latency(us) | static(W) | dynamic(W) | power(W) |
|--------------|---------------|---------------|------|-----|-----------|------------------|-------------|-----------|------------|----------|
| 4096         | 14271(1.59%)  | 18132(1.01%)  | 3    | 12  |314        | 10.039           | 6.549       | 4.406     | 3.779      | 8.185    |
| 2048         | 12446(1.38%)  | 15482(0.86%)  | 3    | 11  |294        | 9.346            | 3.506       | 4.406     | 3.609      | 8.015    |
| 1024         | 12027(1.34%)  | 14953(0.83%)  | 3    | 10  |302        | 9.664            | 1.718       | 4.406     | 3.439      | 7.845    |
| 512          | 11540(1.28%)  | 14489(0.81%)  | 3    | 9   |317        | 10.076           | 0.813       | 4.406     | 3.269      | 7.675    |



