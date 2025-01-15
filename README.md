# Defects4DeepNav Catalog

<p float="middle">
  <img src="rosbotXL/docs/image/IMG_9067.jpg" width="250" />
  <img src="rosbotXL/docs/image/dataset.png" width="250" /> 
  <img src="rosbotXL/docs/image/rosbot-heroimage-new.jpg" width="250" /> 
</p>



In the rapidly evolving field of robotics, deep neural networks (DNNs) have become integral to the navigation systems of autonomous ground robots. However, the complexity of these networks introduces potential points of failure that can lead to significant operational challenges. This  paper introduces a comprehensive repository, Defects4DeepNav, specifically designed to catalog and analyze failures caused by faulty navigation DNNs in ground robots.
Our objective is to systematically identify, document, and summarize instances where commonly used network architectures and training strategies for navigation DNNs have failed, with an emphasis on enabling the analysis of these failures without requiring access to robots.
By curating a diverse set of failure cases, Defects4DeepNav will serve as a critical resource for researchers and engineers seeking to improve testing methods for the reliability and robustness of autonomous navigation systems.
The repository will facilitate the development of diagnostic tools and runtime monitoring tools for robot software, driving forward the state of the art in safe and reliable robotic navigation.


This repository contains automated installation of the Defects4DeepNav catalog and analysis tools to inspect the failures, as well as training scripts for common pytorch implmentations of navigation neural network architectures and options to use common .
It also contains instructions and source code for using the ROSbot to collect a dataset by driving around using the Husarion ROSbot XL.
It contains instructions to install, troubleshoot, and initialize the onboard data collection routine for the ROSbot.

## Catalog Summary

| Model ID | Architecture     | Data Aug. \& Balancing | Inputs  | Dataset Size | Normalization | Epochs       | Loss Function  | Failures |
| -------- | ---------------- | ---------------------- | ------- | ------------ | ------------- | ------------ | -------------- | -------- |
| M1       | DAVE2            | :heavy_check_mark:     | 1       | 11.4K        | :x:           | 100          | MSE            | 13       |
| M2       | DAVE2            | :heavy_check_mark:     | 1       | 11.4K        | :x:           | 100          | L1             | 4        |
| M3       | DAVE2            | :x:                    | 1       | 11.4K        | ImageNet Norm | 100          | MSE            | 12       |
| M4       | DAVE2            | :x:                    | 1       | 11.4K        | BatchNorm     | 100          | MSE            | 12       |
| M5       | DAVE2            | :heavy_check_mark:     | 1       | 11.4K        | :x:           | convergence  | L1             | 2        |
| M6       | MiniTransformer1 | :heavy_check_mark:     | 1       | 11.4K        | :x:           | 100          | MSE            | 12       |
| M7       | MiniTransformer1 | :heavy_check_mark:     | 1       | 11.4K        | :x:           | convergence  | MSE            |          |
| M8       | MiniTransformer1 | :heavy_check_mark:     | 1       | 11.4K        | BatchNorm     | 100          | MSE            |          |
| M9       | DAVE2            | :heavy_check_mark:     | 1       | 30.5K        | :x:           | 100          | MSE            | 12       |
| M10      |             |      | 1       |         |            |           |             |        |
| M11      |             |      | 1       |         |            |           |             |        |
| M12      |             |      | 1       |         |            |           |             |        |
| M13      |             |      | 1       |         |            |           |             |        |
| M14      |             |      | 1       |         |            |           |             |        |
| M15      |             |      | 1       |         |            |           |             |        |

## Quick Links

**Failure catalog:** https://drive.google.com/drive/folders/1Lntd0lctZ05JxOc6pdGFCkYDMpMbI_cN?usp=sharing\
Failures are organized by pretrained model id (M\#, see Table 1 of [failures.pdf](./failures.pdf)). Each model has multiple failures, and each failure has a set of images and a csv file with timestamp, stereo and depth images, LiDAR reading, velocity, DNN prediction, and IMU data.
The name of each failure corr

**Supplemental Demo Video** [via youtube](https://youtu.be/qgvO_J_3u14)\

**Extended failure tables:** [failures.pdf](./failures.pdf)\
This .pdf includes a table of pretrained models for which failures have been collected, and a table for each one of unique failures with a description, small sample trace, and image of the failure circumstances.

**Pretrained models:** https://drive.google.com/drive/folders/1lTqEC30yBuqN6IobSV73E97OaOnCGqDg?usp=sharing\
This project includes a set of pretrained models for ROSbot navigation that can be deployed on the ROSbot XL. It is a superset of all the models for which failures have already been collected.

**All pretrained models:** [all-trained-models.pdf](./all-trained-models.pdf)\
This pdf lists all pretrained models trained using the existing ROSbot navigation dataset. Models 1-6 have been deployed and failures have been collected for the catalogue.


**Dataset:** https://drive.google.com/drive/folders/1Zn7ZNDpPpw7ffnotwR8Jb-DGITRNZy0A?usp=sharing\
The training dataset for all the pretrained models on the 4th floor of Rice Hall at Univrsity of Virginia collected by 4 different drivers under various lighting and hallway configurations and pedestrian traffic conditions.

**Model Archs and training scripts:** [navigation_models/](navigation_models/)\
The code for training these models can be found in the ``navigation_models`` directory.

**ROSbot XL documentation, data collection code, and deployment code:** [rosbotXL](rosbotXL)\
The code, documentation, and troubleshooting guide for ROSbot XL setup, data collection, and deployment of pretrained models can be found in the ``rosbotXL`` directory.


## Directory structure

```bash
.
├── install.sh # setup project environment
├── download.sh # download dataset, pretrained models, and failure catalog
├── README.md 
├── requirements.txt # python environment reqs
├── analysis/ # tools to inspect failure catalog data
│   ├── example_monitors.py
│   ├── query.py
│   ├── replay.py
│   └── summarize.py
├── navigation_models/ # DNN training and validation tools
│   ├── data_cleaning/
│   ├── data_graphing/
│   ├── model_inference/
│   ├── models/
│   └── training/
└── rosbotXL/ # robot data collection and deployment tools and documentation
    ├── docs/
    └── src/
```


## Student offshoot projects from Fall 2023
- [RosBot 2.0](https://github.com/Taylucky/Rosbot2.0)
- [RosBot XL](https://github.com/ish-gupta/ml-robot)
