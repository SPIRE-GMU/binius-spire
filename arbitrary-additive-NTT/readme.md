# Configuration 

Ubuntu 24.04

Vitis 2025.2

Power Design Management 2025.2 (PDM)

# Performance of proposed ANTT architecture in different lengths.

The following table lists the resource consumption, latency, and power consumption of the proposed ANTT architecture in different lengths. 

Some detailed numbers differ due to the Vitis' built-in flow of compilation and implementation, such as resource consumption and frequency. More agressive configuration can be found in "Vitivs -> setting -> xxx.cfg", which may help to further improve the numbers, but that is unnecessary, as the resource utilization are within 2% compared to the on-chip resource available. Same to the timing setting.

The key advantage of the proposed design is that, it has the capability to maintain the throughput around 10 Gb/s in arbitrary length of the ANTT. Moreover, it is a general-purpose architecture, meaning it is appliable to other applications with different length, by adding or removing permutation unit (PU) and butterfly unit (BU) kernels.




| length | LUT(80K)       | FF(1.8M)       | BRAM | Freq(MHz) | Throughput(Gb/s) | Latency(us) | static(W) | dynamic | power(W) |
| :----- | :------------- | :------------- | :--- | :-------- | :--------------- | :---------- | :-------- | :------ | :------- |
| 4096   | 10290(1.14%)   | 11642 (0.65%)  | 3    | 320       | 10.08            | 6.5         | 4.406     | 3.948   | 8.354    |
| 2048   | 12446(1.38%)   | 15482 (0.86%)  | 3    | 314       | 10.458           | 3.133       | 4.406     | 3.609   | 8.015    |
| 1024   | 12027(1.34%)   | 14953 (0.83%)  | 3    | 302       | 10.297           | 1.591       | 4.406     | 3.439   | 7.845    |
| 512    | 11540(1.28%)   | 14489 (0.81%)  | 3    | 317       | 10.002           | 0.819       | 4.406     | 3.269   | 7.675    |




# Screen shot

Here are some screen shots captured during the work, which may help to understand the work. 
