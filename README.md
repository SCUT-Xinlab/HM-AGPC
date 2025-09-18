# HM-AGPC

[MICCAI 2025]The official code for Heterogeneous Masked Attention-Guided Path Convolution for Functional Brain Network Analysis

![The overview of HM-AGPC. Step 1 (blue): Functional Brain Network Construction. Step 2 (orange): Masked Attention Generation and Path Convolution, including our two key components: Heterogeneous Masked Attention Generation and Attention Guided Path Convolution. Step 3 (red): Readout and Prediction.](figure/main3.png)

The overview of HM-AGPC. Step 1 (blue): Functional Brain Network Construction. Step 2 (orange): Masked Attention Generation and Path Convolution, including our two key components: Heterogeneous Masked Attention Generation and Attention Guided Path Convolution. Step 3 (red): Readout and Prediction.

![The diagram of Heterogeneous Masked Attention Generation (yellow) and  Attention-Guided Path Convolution (green). The heterogeneous and homogeneous pathways are marked with red and blue respectively.](figure/block7.png)

The diagram of Heterogeneous Masked Attention Generation (yellow) and  Attention-Guided Path Convolution (green). The heterogeneous and homogeneous pathways are marked with red and blue respectively.

## Instructions
HM_AGPC is our main model.
self.node_list is the partition prior for AAL atlas, should be changed based on the chosen atlas and prior partitions.
train_demo.py provide a training demo. train_data and test_data should be changed to real-world datas.
ABIDE dataset can be download from [here](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html). 

## **Code environment**
python 3.9
pytorch 2.0.1
