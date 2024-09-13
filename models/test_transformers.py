# https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
# https://github.com/google-research/vision_transformer
from torchvision.models.vision_transformer import *
import torchvision
import torch
from torchvision.io import read_image, ImageReadMode

M2 = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
print(M2)
im = read_image("/p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_data_off_course_backup/collection000/rivian-00137.jpg", mode=ImageReadMode.RGB)
print(im.shape)
# NICE ORGANIZED CLEAN: https://github.com/JDScript/COMP3340-gp/tree/74b80c482420420e4f21aa55c512c07477712cd4


# https://github.com/lucidrains/vit-pytorch?tab=readme-ov-file#vision-transformer---pytorch

# BACKUP:
# https://github.com/tahmid0007/VisionTransformer/blob/main/Google_ViT.py
# model_it = ImageTransformer(image_size=32, patch_size=4, num_classes=10, channels=3,
#             dim=64, depth=6, heads=8, mlp_dim=128)