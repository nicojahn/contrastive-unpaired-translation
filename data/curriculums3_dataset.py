import numpy as np

from data.base_dataset import BaseDataset, get_transform
from data.unaligneds3_dataset import UnalignedS3Dataset


class CurriculumS3Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument("--curriculum_transition", nargs=2, metavar=("first epoch of transition", "last epoch of transition"), type=int, required=True, help="The exclusive epochs which state the transition period between two datasets")
        parser.add_argument("--curriculum_first_paths", nargs=2, metavar=("bucket/path", "bucket/path"), type=str, required=True, help="")
        parser.add_argument("--curriculum_second_paths", nargs=2, metavar=("bucket/path", "bucket/path"), type=str, required=True, help="")
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.transition = sel.opt.curriculum_transition
        assert self.transition[0] > 0 and self.transition[1] > 0, "Transition epochs need to be positive (>0)."
        self.difference = self.transition[1]-self.transition[0]
        assert self.difference > 0, "As the transition boundaries are exclusive, you need at least a difference of 1 epoch."

        A, B = self.opt.curriculum_first_paths
        self.d1 = UnalignedS3Dataset(opt, s3_path_A= "s3://" + A, s3_path_B= "s3://" + B)
        A, B = self.opt.curriculum_second_paths
        self.d2 = UnalignedS3Dataset(opt, s3_path_A= "s3://" + A, s3_path_B= "s3://" + B)
        self.datasets = [self.d1, self.d2]
        
        self.random_dataset = lambda epoch: np.random.choice([0, 1], p=tuple([np.abs(-1+idx+p) for idx, p in enumerate(2*[(epoch-self.transition[0])/self.difference])]))
        self.current_dataset = lambda epoch: self.d1 if epoch <= self.transition[0] else self.d2 if epoch >= self.transition[1] else self.datasets[self.random_dataset(epoch)]

    def __getitem__(self, index):
        d = self.current_dataset(self.current_epoch)
        return d.__getitem__(index)

    def __len__(self):
        n = 1000
        # empirical length (mean over n draws)
        length = np.round(np.mean(list(map(lambda _: len(self.current_dataset(self.current_epoch)), range(n)))))
        return int(length)
