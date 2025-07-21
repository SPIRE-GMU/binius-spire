# Binius-spire

This is the repository for Binius project, we will share our progress here.


### /ref 

This folder includes reference python code of Binius. 

Having profilied the whole Binius scheme, we find the botlenecks lies in function extend(). To be more specific, that is additive-ntt (inver_addi_ntt) which is most time consuming component. By profiling the code on i9-14900K CPU core, it takes 98.173 second to execute extend() function. 

Hence, this project focuses on accelerating function extend() in the ref code as a first step, before implementing the complete Binius on VCK-190.    

### Configuration of arbitrary-additive-NTT on VCK:

For AIE-component, set stack size to 10240

For PL kernel, set Freq. to 512 MHz. (Maybe higher is allowed but remains test)

