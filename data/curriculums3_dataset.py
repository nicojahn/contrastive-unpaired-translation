import numpy as np

from data.base_dataset import BaseDataset, get_transform
from data.unaligneds3_dataset import UnalignedS3Dataset


class CurriculumS3Dataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # TODO: Add this to command line options
        self.transition = (2,7)
        # TODO: Add environment variable prefix to command line options (for BOTH)
        self.d1 = UnalignedS3Dataset(opt)
        self.d2 = UnalignedS3Dataset(opt)

        assert self.transition[0] > 0 and self.transition[1] > 0, "Transition epochs need to be positive (>0)."
        self.difference = self.transition[1]-self.transition[0]
        assert self.difference > 0, "As the transition boundaries are exclusive, you need at least a difference of 1 epoch."
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
