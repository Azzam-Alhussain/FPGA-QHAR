# Quantized Human Action Recognition on edge AMD SoC-FPGAs

### [Azzam Alhussain](http://azzam.page/), [Mingjie Lin](https://www.ece.ucf.edu/person/mingjie-lin/)
___
**This is the official efficient real-time HW/SW Co-design for quantized two-stream CNN of Human Action Recognition (FPGA-QHAR) on PYNQ SoC-FPGAs that is accepted as a conference paper in the IEEE Xplore Digital Library as [FPGA-QHAR: Throughput-Optimized for Quantized Human Action Recognition on The Edge](https://arxiv.org/abs/2311.03390), and will be presented in December 2023 at the [IEEE 20th International Conference on SmartCommunities: Improving Quality of Life Using AI, Robotics and IoT](https://honet-ict.org/index.html).**

## Description

This paper proposed an end-to-end efficient customized and quantized Two-Stream HAR [SimpleNet-PyTorch CNN](https://github.com/Coderx7/SimpleNet_Pytorch) architecture trained on [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) & [UCF24](https://github.com/gurkirt/realtime-action-detection/blob/master/data/ucf24.py) datasets and implemented as HW/SW co-design on AMD PYNQ SoC-FPGAs using partially streaming dataflow architecture that achieved real-time performance of 24FPS with 81% prediction accuracy on connected camera. 


![image](https://github.com/Azzam-Alhussain/FPGA-QHAR/assets/74447207/aeb8c0ca-7bd4-4fea-a0e7-a2e245b738ba)


## Contributions
- Developed a scalable inference accelerator for transpose convolution operation for quantized DCGAN (QDCGAN) on top of [FINN by Xilinx](https://xilinx.github.io/finn/). 
- Provided a complete open-source framework (training to implementation stack) for investigating the effect of variable bit widths for weights and activations. 
- Demonstrated that the weights and activations influence performance measurement, resource utilization, throughput, and the quality of the generated images.
- The community can build upon our code, explore, and search efficient implementation of SRGAN on low-power FPGAs which are considered as a solution for a wide range of medical   and microscopic imaging applications.
The developed network accelerator fused all convolutional, batch-norm, and ReLU operations into a single homogeneous layer and utilized the Lucas-Kanade motion flow method to enable an optimized on-chip engine computing on FPGA, while GPU, CPU, and Jetson don't have this capability.  

## Getting Started

### Requirement
* Nvidia GPU
* Linux Ubuntu 18.04
* Python 3.6+
* Pytorch 1.4.0+
* Vivado 2019.3+ 
* PYNQ framework 2.6
* Xilinx SoC-FPGAs Pynq supported (ex: Ultra96 & ZCU104)

### HW/SW training & implementation

- `PyTorch` folder for training.
- `Hardware` folder for the synthesis of the accelerator.
- `Hardware/Pynq/` folder for deployment on xilinx SOC-FPGAs having pynq linux.

## License

All source code is made available under a BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is currently accepted, and will be published soon as a conference paper in the IEEE Xplore Digital Library.

## Citation

Please use the below citation till it updated from IEEE Xplore Digital Library,

**Alhussain, A. and Lin, M, "Hardware-Efficient Deconvolution-Based GAN for Edge Computing," in 56th Annual Conference on Information Sciences and Systems (CISS) 2022, Virtual IEEE conference, March 9-11, 2022. pp.1-5, Accessed on: Jan. 19, 2022. [Online]. Available: https://arxiv.org/abs/2201.06878**

## Acknowledgments

Inspiration, code snippets, references, etc.

* [FINN: A Framework for Fast, Scalable Binarized Neural Network Inference](https://xilinx.github.io/finn/)
* [Xilinx/finn-hlslib](https://github.com/Xilinx/finn-hlslib)
* [Xilinx/brevitas](https://github.com/Xilinx/brevitas)
* [Xilinx/PYNQ](https://github.com/Xilinx/PYNQ)
* [Xilinx/BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ)
* [A Competitive Edge: Can FPGAs Beat GPUs at DCNN Inference Acceleration in Resource-Limited Edge Computing Applications?](https://arxiv.org/pdf/2102.00294v2.pdf)
