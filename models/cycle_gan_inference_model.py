from .inference_model import InferenceModel
from .cycle_gan_model import CycleGANModel
from options.train_options import TrainOptions        
import argparse


class CycleGANInferenceModel(CycleGANModel, InferenceModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CycleGANModel.modify_commandline_options(parser, is_train=True)
        parser = argparse.ArgumentParser(parents=[TrainOptions().initialize(argparse.ArgumentParser()), parser], conflict_handler='resolve')
        return parser

    def discriminate(self):
        self.idt_A = self.netG_A(self.real_B)
        self.idt_B = self.netG_B(self.real_A)
        return self.netD_A(self.real_B), self.netD_A(self.fake_B)