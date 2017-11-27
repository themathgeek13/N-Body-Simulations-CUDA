Authors: Rohan Rao (EE14B118)
		 Akshun Yadav (EE14B070)

CPU & Rendering code source: https://github.com/PWhiddy/Nbody-Gravity
(this code was written purely for CPU, and we tried to improve it by optimizing for GPU using various techniques)

Folder Structure:
AllPairs_N2 -------> BarnzNhutt_n2threads.cu
				|
				|--> BarnzNhutt_optimal.cu
				|
				|--> BarnzNhutt_tiled.cu

DynamicPar_NlogN---> cudanlogn.cu

Instructions:
1. Go to the folder of interest and modify the "build.bash" file and constants in the "Constants.h" file as per the requirement.
2. Run build.bash and then ./a.out
3. Stop after the required number of iterations.
4. Use the "createVideo.bash" file to use ffmpeg and convert the ppm images into a .mp4 video file
5. Delete the images using the "deleteImgs.bash" file

Other Resources:
1) https://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/
2) http://on-demand.gputechconf.com/gtc/2012/presentations/S0338-GTC2012-CUDA-Programming-Model.pdf
3) https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch31.html

Challenges:
1) Maximize coalescing and optimize memory access on device
2) Minimize CPU/GPU data transfer, keep data on GPU between kernel calls
3) Use of Unified Virtual Memory (UVM) to implement deepcopy of linked tree structures
4) Use of dynamic parallelism for recursive NlogN tree building and force computation
5) Use of streams within parent DP kernels for ensuring concurrent execution of child kernels
