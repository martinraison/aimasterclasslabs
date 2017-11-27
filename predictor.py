import torch
import models

class Predictor:

    def __init__(self):
        ckpt = torch.load("model.pth")
        self.model = ckpt['net'].eval()

    def __call__(self, data):
        return self.model(data)
