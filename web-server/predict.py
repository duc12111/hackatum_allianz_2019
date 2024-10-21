import torch
from torch import Tensor
import numpy as np
import cv2
import sys

sys.path.insert(0, '/Users/ducnguyen/Desktop/duc doc/hackatum-allianz/segmentation_models.pytorch')
import segmentation_models_pytorch as smp

model_path = "../segmentation_models.pytorch/challenge/best_model.pth"


def predict(img):
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    model = torch.load(model_path)
    img = preprocessing_fn(img)
    tensor_img = Tensor(img).to("cuda")
    tensor_img = tensor_img.unsqueeze(dim=0)
    tensor_img = tensor_img.transpose(1, 3)
    tensor_img = tensor_img.transpose(2, 3)
    result = model.predict(tensor_img)
    result = result.squeeze()
    return result.cpu().numpy()
