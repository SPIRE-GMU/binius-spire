# Configuration 

Ubuntu 24.04

Vitis 2025.2

Power Design Management 2025.2 (PDM)

# Performance of proposed ANTT architecture in different lengths.

The following table lists the resource consumption, latency, and power consumption of the proposed ANTT architecture in different lengths. 

The key advantage of the proposed design is that, it has the capability to maintain the throughput around 10 Gb/s in arbitrary length of the ANTT. Moreover, it is a general-purpose architecture, meaning it is appliable to other applications involving different length, by adding or removing permutation unit (PU) and butterfly unit (BU) kernels. Meanwhile, the resource and power consumption changes accordingly.

Noting, we provide the table as a reference but it may not obtain the exactly same numbers when reproducing the framework due to the Vitis' built-in compilation flow.  



| Length       | LUT(80K)      | FF(1.8M)      | BRAM | Freq(MHz) | Throughput(Gb/s) | Latency(us) | static(W) | dynamic(W) | power(W) |
|--------------|---------------|---------------|------|-----------|------------------|-------------|-----------|------------|----------|
| 4096         | 14271(1.59%)  | 18132(1.01%)  | 3    | 314       | 10.039           | 6.549       | 4.406     | 3.779      | 8.185    |
| 2048         | 12446(1.38%)  | 15482(0.86%)  | 3    | 294       | 9.346            | 3.506       | 4.406     | 3.609      | 8.015    |
| 1024         | 12027(1.34%)  | 14953(0.83%)  | 3    | 302       | 9.664            | 1.718       | 4.406     | 3.439      | 7.845    |
| 512          | 11540(1.28%)  | 14489(0.81%)  | 3    | 317       | 10.076           | 0.813       | 4.406     | 3.269      | 7.675    |



