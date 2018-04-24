# Progressive refinement: a method of coarse-to-fine image parsing using stacked network

This repository contains the code for the stacked network framework for image parsing introduced in the following paper

[Progressive refinement: a method of coarse-to-fine image parsing using stacked network](https://arxiv.org/abs/1804.08256) (ICME 2018)

### Citation
If you find this work is useful in your research, please consider citing:

	@inproceedings{hu2018stacked,
	  title={Progressive refinement: a method of coarse-to-fine image parsing using stacked network},
	  author={Hu, Jiagao and Sun, Zhengxing and Sun, Yunhan and Shi, Jinglong },
	  booktitle={2018 {IEEE} International Conference on Multimedia and Expo, {ICME} 2018, San Diego, USA, July 23-27, 2018},
	  year={2018}
	}
	

## Introduction
In this paper, a coarse-to-fine image parsing framework is proposed to parse images progressively with refined semantic classes by stacking several FCNs. The first network is trained to segment images at a coarse-grained level, and the last one is trained at the finest-grained level. To remove the redundant computation in stacked networks, we propose to share the image encoding parts (i.e., the former layers) of each network, which results in a standard FCNs with multiple stacked segmentation modules. To parse the fine-grained semantic parts precisely, we add skip connections from shallow layers which can provide more structural and localization details to the fine-grained segmentation modules. For the network training, we merge some classes in the finest-grained groundtruth label map to get a set of coarse-to-fine label maps, and train the network with this hierarchical supervision. The stacked network can be trained end-to-end and can get progressively refined hierarchical parsing result in a single forward pass. In addition, the coarse-to-fine stacking strategy can be injected into many advanced image segmentation networks for image parsing tasks. The following figure shows the framework of our stacked networks.

![Image](/images/framework.png "Framework of our stacked network ")