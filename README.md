# Binius-spire

This is the repository for Binius project, we will share our progress here.


### Reference code 

Folder /ref includes reference python code of Binius. Source: https://github.com/ethereum/research/tree/master/binius

Having profilied the whole Binius scheme, we find the botlenecks lies in function extend(). To be more specific, that is additive-ntt (inver_addi_ntt) which is most time consuming component. By profiling the code on i9-14900K CPU core, it takes 98.173 second to execute extend() function. 

Hence, this project focuses on accelerating additive_ntt (inverse_additive_ntt) in the ref code as a first step, before implementing the complete Binius on VCK-190.  

The implementation is 4096-point NTT but remember the architecture is scalable for NTT in arbitrary length. 

Golden samples as well as coefficients are printed for convenient debug.


We also refer to kernels in https://github.com/ingonyama-zk/open-binius for implementation in PL domain.

### Tools:

OS: Ubuntu 2024.2

Vitis 2025.1 (other version is ok but may differ in details.)

### Configuration of arbitrary-additive-NTT on VCK:


For AIE-component, set stack size to 10240

For PL kernel, set Freq. to 512 MHz. (Maybe higher is allowed but remains test)

