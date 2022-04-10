from .inference_model import InferenceModel
from .cut_model import CUTModel
from options.train_options import TrainOptions
import argparse


class CUTInferenceModel(CUTModel, InferenceModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train=True)
        parser = argparse.ArgumentParser(parents=[TrainOptions().initialize(argparse.ArgumentParser()), parser], conflict_handler='resolve')
        return parser

    def discriminate(self):
        return self.netD(self.real_B), self.netD(self.fake_B)