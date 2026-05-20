import torch
import torch.nn as nn
from models.team07_DVMSR import DVMSR


# ================= ILLUMINATION =================
class IlluminationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,3,3,1,1), nn.Sigmoid()
        )

    def forward(self,x):
        return x * self.net(x)


# ================= FULL MODEL =================
class FullModel(nn.Module):
    def __init__(self, use_illum=False):
        super().__init__()

        self.use_illum = use_illum
        self.dvmsr = DVMSR()

        if self.use_illum:
            self.illum = IlluminationModule()

    def forward(self,x):
        if self.use_illum:
            x = self.illum(x)
        return self.dvmsr(x)


# ================= FACTORY =================
def get_model(model_type):
    """
    Central place for model creation
    """

    if model_type in ["baseline", "gamma", "perceptual", "gamma_perceptual"]:
        return FullModel(use_illum=False)

    elif model_type in ["illum", "full"]:
        return FullModel(use_illum=True)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")