from .cut_model import CUTModel
from .cycle_gan_model import CycleGANModel
import argparse

# TODO: Decide upon some option
use_cut = True

class InferenceModel(CUTModel if use_cut else CycleGANModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        if use_cut:
            parser = CUTModel.modify_commandline_options(parser, is_train=True)
        else:
            parser = CycleGANModel.modify_commandline_options(parser, is_train=True)

        from options.train_options import TrainOptions        
        parser = argparse.ArgumentParser(parents=[TrainOptions().initialize(argparse.ArgumentParser()), parser], conflict_handler='resolve')
        return parser

    def __init__(self, opt):
        opt.isTrain = True
        if use_cut:
            CUTModel.__init__(self, opt)
        else:
            CycleGANModel.__init__(self, opt)
        opt.isTrain = False
        self.isTrain = False

    def forward(self):
        if use_cut:
            CUTModel.forward(self)
            discriminate = lambda x: self.netD(x)
        else:
            CycleGANModel.forward(self)
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)
            discriminate = lambda x: self.netD_A(x)
        
        # networks.py: GANLoss.__init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0)
        discriminated_real_B = discriminate(self.real_B)
        discriminated_fake_B = discriminate(self.fake_B)
        # discriminated_idt_B = discriminate(self.idt_B)

        # RMSE
        rmse = ((discriminated_real_B-discriminated_fake_B)**2).mean()**0.5
        print(f"{self.image_paths}: {rmse}")
