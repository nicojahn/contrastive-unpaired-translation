from data.base_dataset import BaseDataset, get_transform

from awsio.python.lib.io.s3.s3dataset import S3Dataset as S3DS
import os
from PIL import Image
import io
import random
import util.util as util


class UnalignedS3Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument("--s3_path_A", type=str, required=True, help="")
        parser.add_argument("--s3_path_B", type=str, required=True, help="")
        parser.add_argument("--grayscale", action="store_true", help="")
        # Using as: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Resampling
        parser.add_argument("--rescale_method", type=str, default="BICUBIC", choices=["NEAREST", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS"], help="")
        return parser

    def __init__(self, opt, s3_path_A=None, s3_path_B=None):
        """Initialize this dataset class. Creating two S3 dataset instances with different paths (or potentially S3 Buckets)

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if s3_path_A is None:
            s3_path_A = "s3://" + self.opt.s3_path_A
        if s3_path_B is None:
            s3_path_B = "s3://" + self.opt.s3_path_B

        self.S3SourceDomain = S3DS(s3_path_A)
        self.S3TargetDomain = S3DS(s3_path_B)
        self.A_size = len(self.S3SourceDomain)
        self.B_size = len(self.S3TargetDomain)

        self.grayscale = self.opt.grayscale
        self.method = getattr(Image.Resampling, self.opt.rescale_method.upper())

    def __getitem__(self, index):
        """Basically the same as in 'unaligned_dataset.py'. But instead reading from paths, reading from S3 datasets.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        # The following is almost the same as in 'unaligned_dataset.py'
        index_A = index % self.A_size
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)

        # begin: different
        A_path, A_img = self.S3SourceDomain.__getitem__(index_A)
        B_path, B_img = self.S3TargetDomain.__getitem__(index_B)
        A = Image.open(io.BytesIO(A_img)).convert("RGB")
        B = Image.open(io.BytesIO(B_img)).convert("RGB")
        # end: different

        # The following is almost the same as in 'unaligned_dataset.py'
        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        #        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(
            self.opt,
            load_size=self.opt.crop_size if is_finetuning else self.opt.load_size,
        )
        transform = get_transform(modified_opt, grayscale=self.grayscale, method=self.method)
        A = transform(A)
        B = transform(B)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    # The following is the same as in 'unaligned_dataset.py'
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
