# ROSbot Dataset Collection

<p float="middle">
  <img src="figures/IMG_8418.jpg" width="250" />
  <img src="figures/dataset.png" width="250" /> 
</p>
<!-- | ![](figures/IMG_8418.jpg)  |  ![](figures/dataset.jpg) | -->

This repository contains instructions and source code for using the ROSbot to collect a dataset by driving around using the Husarion ROSbot 2.0.
It contains instructions to install, troubleshoot, and initialize the onboard data collection routine for the ROSbot.
It also contains training scripts for a DAVE2 steering model and pytorch implmentations of other architectures.

```python
# documentation
docs
├── data_collection_quickstart.md
├── rosbot_basics.md
├── rosbot_installation.md
├── rosbot_usage.md
├── ROSbot-applications.txt
├── ROSbot-pub-topics.txt
└── datacoll
# data collection
start_rosbot.sh
src
├── CMakeLists.txt -> /opt/ros/noetic/share/catkin/cmake/toplevel.cmake
└── datacoll
    ├── CMakeLists.txt
    ├── launch
    │   └── data_collector.launch
    ├── package.xml
    └── src
        ├── dataset_writer.py
        └── teleop_joy_concurrent.py
# training
models/
├── DAVE2pytorch.py
├── README.md
├── ResNet.py
└── VAE.py
training/
├── DatasetGenerator.py
├── install.sh
├── requirements.txt
├── README.md
└── train_DAVE2.py
```
