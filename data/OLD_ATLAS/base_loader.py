import torchvision.transforms as transforms

from core.DataLoader import DefaultDataset
from transforms.preprocessing import *




class Flip:
    """
    Flip brain

    """

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        return torch.tensor((img.astype(np.float32)).copy())


class BaseLoader(DefaultDataset):
    def __init__(
        self,
        data_dir,
        file_type="",
        label_dir=None,
        mask_dir=None,
        target_size=(256, 256),
        test=False,
    ):
        self.target_size = target_size
        self.RES = transforms.Resize(self.target_size)
        super(BaseLoader, self).__init__(
            data_dir, file_type, label_dir, mask_dir, target_size, test
        )

    def get_image_transform(self):
        default_t = transforms.Compose(
            [
                ReadImage(),
                To01(),  # , Norm98(),
                Pad((1, 1)),  # Flip(), #  Slice(),
                AddChannelIfNeeded(),
                AssertChannelFirst(),
                self.RES,
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )
        return default_t

    def get_image_transform_test(self):
        default_t_test = transforms.Compose(
            [
                ReadImage(),
                To01(),  # , Norm98()
                Pad((1, 1))
                # Flip(), #  Slice(),
                ,
                AddChannelIfNeeded(),
                AssertChannelFirst(),
                self.RES,
            ]
        )
        return default_t_test

    def get_label_transform(self):
        default_t_label = transforms.Compose(
            [
                ReadImage(),
                To01(),
                Pad((1, 1)),
                AddChannelIfNeeded(),
                AssertChannelFirst(),
                self.RES,
            ]
        )  # , Binarize()])
        return default_t_label
