# Quantized Human Action Recognition on edge AMD SoC-FPGAs

### [Azzam Alhussain](http://azzam.page/), [Mingjie Lin](https://www.ece.ucf.edu/person/mingjie-lin/)
___
**This is the official efficient real-time HW/SW Co-design for quantized two-stream CNN of Human Action Recognition (FPGA-QHAR) on PYNQ SoC-FPGAs that is accepted as a conference paper in the IEEE Xplore Digital Library as [FPGA-QHAR: Throughput-Optimized for Quantized Human Action Recognition on The Edge](https://arxiv.org/abs/2311.03390), and will be presented in December 2023 at the [IEEE 20th International Conference on SmartCommunities: Improving Quality of Life Using AI, Robotics and IoT](https://honet-ict.org/index.html).**

## Description & System Overview 

This paper proposed an end-to-end efficient customized and quantized Two-Stream HAR [SimpleNet-PyTorch CNN](https://github.com/Coderx7/SimpleNet_Pytorch) architecture trained on [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) & [UCF24](https://github.com/gurkirt/realtime-action-detection/blob/master/data/ucf24.py) datasets and implemented as HW/SW co-design on AMD PYNQ SoC-FPGAs using partially streaming dataflow architecture that achieved real-time performance of 24FPS with 81% prediction accuracy on connected camera. 


![image](https://github.com/Azzam-Alhussain/FPGA-QHAR/assets/74447207/aeb8c0ca-7bd4-4fea-a0e7-a2e245b738ba)


## Contributions

![Fig2 drawio (4)](https://github.com/Azzam-Alhussain/FPGA-QHAR/assets/74447207/e2f9f7b2-975d-47c0-83ae-02c5232fb972)


- Developed a scalable inference accelerator for QHAR on top of [SimpleNet-PyTorch CNN](https://github.com/Coderx7/SimpleNet_Pytorch) & [NetDBFPGA](https://github.com/NetDBFPGA/ecv2021_demo/tree/master).
- The developed network accelerator fused all convolutional, batch-norm, and ReLU operations into a single homogeneous layer and utilized the Lucas-Kanade motion flow method to enable an optimized on-chip engine computing on FPGA, while GPU, CPU, and Jetson don't have this capability.  
- Provided a complete open-source framework (training to implementation stack) for QHAR on SoC-FPGA and different hardware platforms. 
- Demonstrated that the small version of UCF101 which is [UCF24](https://github.com/gurkirt/realtime-action-detection/blob/master/data/ucf24.py) datasets effect positively the performance & accuracy, resource utilization, and throughput.
- The community can build upon our code, explore, and search efficient implementation of Multimodal fusion for comprehensive ADAS with HAR action understanding on low-power FPGAs which are considered as a solution for a wide range of Autonomous applications.

## Getting Started

### Requirement
* Local Nvidia GPU or [Google Cloud GPU with Colab](https://colab.research.google.com/)
* Linux Ubuntu 18.04
* Python 3.7+
* Pytorch v1.12.0+
* Vivado 2018.3 
* PYNQ framework 2.6
* AMD SoC-FPGAs Pynq supported (ex: Kria KV260 & ZCU104)

### HW/SW training & implementation

- `PyTorch` folder for training.
- `HLS` folder for the synthesis of the accelerator.
- `PYNQ_Hardware` folder for deployment on xilinx SOC-FPGAs having pynq linux.

## License

All source code is made available under a BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is currently accepted, and will be published soon as a conference paper in the IEEE Xplore Digital Library.

## Citation

**TBD** 

## Acknowledgments

Inspiration, code snippets, references, etc.

* [FINN: A Framework for Fast, Scalable Binarized Neural Network Inference](https://xilinx.github.io/finn/)
* [Xilinx/PYNQ](https://github.com/Xilinx/PYNQ)
* [Efficient Two-stream Action Recognition on FPGA](https://github.com/NetDBFPGA/ecv2021_demo/tree/master)
