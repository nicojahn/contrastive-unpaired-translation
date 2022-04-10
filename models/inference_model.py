from .base_model import BaseModel
import numpy as np
import os
from pathlib import Path


class InferenceModel(BaseModel):

    def __init__(self, opt):
        opt.isTrain = True
        super().__init__(opt)
        opt.isTrain = False
        self.isTrain = False
        
        self.path = f"discriminator/{self.opt.name}"
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def forward(self):
        super().forward()
        assert len(self.image_paths) == 1

        # networks.py: GANLoss.__init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0)
        discriminated_real_B, discriminated_fake_B = self.discriminate()

        base_name = Path(self.image_paths[0]).stem
        np.save(f"{self.path}/{base_name}_real.npy", discriminated_real_B.detach().numpy())
        np.save(f"{self.path}/{base_name}_fake.npy", discriminated_fake_B.detach().numpy())

        # RMSE
        # rmse = ((discriminated_real_B-discriminated_fake_B)**2).mean()**0.5
        # print(f"{self.image_paths}: {rmse}")
