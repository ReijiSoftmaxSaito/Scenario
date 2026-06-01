import cv2
import torch
import numpy as np
import torchvision


def show_anomaly(images, labels, anomaly, save_path, limit):

    images, labels, anomaly = images.cpu(), labels.cpu(), anomaly.cpu()
    labels = labels.expand(-1, 3, -1, -1)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean = images*std[None, :, None, None] + mean[None, :, None, None]

    anomaly = (anomaly - limit[0]) / (limit[1] - limit[0] + 1e-8)

    annomaly = anomaly.detach().cpu().numpy() * 255
    annomaly = annomaly.astype(np.uint8)
    anomaly = np.stack([cv2.applyColorMap(ano[0], cv2.COLORMAP_JET) for ano in annomaly])
    anomaly = anomaly.permute(0, 3, 1, 2)[:,[2,1,0]]
    rate = 0.7
    anomaly = images*rate + anomaly*(1-rate)
    groups = torch.cat([images, labels, anomaly], dim=0)

    torchvision.utils_save_image(groups, save_path, nrow=len(images), normalize=True)