import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as transform
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import Transform
from monai.utils.enums import TransformBackends


class ReadImage(Transform):
    """
    Transform to read image, see torchvision.io.image.read_image
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, path: str) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if ".npy" in path:
            # return torch.tensor(np.flipud((np.load(path).astype(np.float32)).T).copy())  # Mid Axial slice of MRI brains
            img = np.load(path).astype(np.float32)
            img = (img * 255).astype(np.uint8)
            return torch.tensor(img)
        elif ".jpeg" in path or ".jpg" in path or ".png" in path:
            PIL_image = PIL.Image.open(path)
            tensor_image = torch.squeeze(transform.to_tensor(PIL_image))
            return tensor_image
        elif ".nii.gz" in path:
            import nibabel as nip
            from nibabel.imageglobals import LoggingOutputSuppressor

            with LoggingOutputSuppressor():
                img_obj = nip.load(path)
                img_np = np.array(img_obj.get_fdata(), dtype=np.float32)
                img_t = torch.Tensor(img_np.copy())
                # img_t = torch.Tensor(np.flipud(img_np[:, :, 95].T).copy()) # Mid Axial slice of MRI brains
            return img_t
        elif ".nii" in path:
            import nibabel as nip

            img = nip.load(path)
            return torch.Tensor(np.array(img.get_fdata()))
        elif ".dcm" in path:
            from pydicom import dcmread

            ds = dcmread(path)
            return torch.Tensor(ds.pixel_array)
        elif ".h5" in path:  ## !!! SPECIFIC TO FAST MRI, CHANGE FOR OTHER DATASETS
            import h5py

            f = h5py.File(path, "r")
            img_data = f["reconstruction_rss"][:]  # Fast MRI Specific
            img_data = img_data[:, ::-1, :][0]  # flipped up down
            return torch.tensor(img_data.copy())
        else:
            raise IOError


class Norm98:
    def __init__(self, max_val=255.0):
        self.max_val = max_val
        super(Norm98, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        # print(torch.min(img), torch.max(img))
        # print(img.shape)
        q = np.percentile(img, 98)
        # q is a pixel value, below which 98% of the pixels lie
        img = img / q
        img[img > 1] = 1
        # return img/self.max_val
        return img


class To01:
    """
    Convert the input to [0,1] scale

    """

    def __init__(self, max_val=255.0):
        self.max_val = max_val
        super(To01, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        if torch.max(img) <= 1.0:
            # print(img.cpu().numpy().shape)
            return img
        # print(img.cpu().numpy().shape)
        if torch.max(img) <= 255.0:
            return img / 255

        return img / 65536
    
class NormalizeRange(Transform):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        super(NormalizeRange, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        q = np.percentile(img, 98)
        clipped = np.clip(img, 0, q)
        img = (clipped - clipped.min()) / (clipped.max() - clipped.min())
        scaled = img * (self.max_val - self.min_val) + self.min_val
        return scaled

class AdjustIntensity:
    def __init__(self):
        self.values = [1, 1, 1, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
        # self.methods = [0, 1, 2]

    def __call__(self, img):
        value = np.random.choice(self.values)
        # method = np.random.choice(self.methods)
        # if method == 0:
        return torchvision.transforms.functional.adjust_gamma(img, value)


class Binarize:
    def __init__(self, th=0.5):
        self.th = th
        super(Binarize, self).__init__()

    def __call__(self, img):
        img[img > self.th] = 1
        img[img < 1] = 0
        return img


class MinMax:
    """
    Min Max Norm
    """

    def __call__(self, img):
        max = torch.max(img)
        min = torch.min(img)
        img = (img - min) / (max - min)
        return img


class ToRGB:
    """
    Convert the input to an np.ndarray from grayscale to RGB

    """

    def __init__(self, r_val, g_val, b_val):
        self.r_val = r_val
        self.g_val = g_val
        self.b_val = b_val
        super(ToRGB, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        r = np.multiply(img, self.r_val).astype(np.uint8)
        g = np.multiply(img, self.g_val).astype(np.uint8)
        b = np.multiply(img, self.b_val).astype(np.uint8)

        img_color = np.dstack((r, g, b))
        return img_color


class AddChannelIfNeeded(Transform):
    """
    Adds a 1-length channel dimension to the input image, if input is 2D
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if len(img.shape) == 2:
            # print(f'Added channel: {(img[None,...].shape)}')
            return img[None, ...]
        else:
            return img


class AssertChannelFirst(Transform):
    """
    Assert channel is first and permute otherwise
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        assert (
            len(img.shape) == 3
        ), f"AssertChannelFirst:: Image should have 3 dimensions, instead of {len(img.shape)}"
        if img.shape[0] == img.shape[1] and img.shape[0] != img.shape[2]:
            print(f"Permuted channels {(img.permute(2,0,1)).shape}")
            return img.permute(2, 0, 1)
        elif img.shape[0] > 1:
            return img[0:1, :, :]
        else:
            return img


class Slice(Transform):
    """
    Pad with zeros
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # x = int(320 - img.shape[0] / 2)
        # y = int(320 - img.shape[1] / 2)
        # self.pid = (x, y)
        mid_slice = int(img.shape[0] / 2)
        img_slice = img[mid_slice, :, :]
        return img_slice

class Zoom(Transform):
    """
    Resize 3d volumes
    """

    def __init__(self, input_size):
        self.input_size = input_size
        self.mode = "trilinear" if len(input_size) > 2 else "bilinear"

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if len(img.shape) == 3:
            img = img[None, ...]
        return F.interpolate(img, size=self.input_size, mode=self.mode)[0]

class AtlasAssertChannelFirst(Transform):
    """
    Assert channel is first and permute otherwise in Atlas dataset
    """

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        assert (
            len(img.shape) == 4
        ), f"AssertChannelFirst:: Image should have 4 dimensions, instead of {len(img.shape)}"
        return img.permute(0, 2, 3, 1)

class AddChannelIfNeeded3D(Transform):
    """
    Adds a 1-length channel dimension to the input image, if input is 2D
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if len(img.shape) == 3:
            # print(f'Added channel: {(img[None,...].shape)}')
            return img[None, ...]
        else:
            return img

class Pad3d(Transform):
    """
    Pad with zeros
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, pad):
        self.pad = pad

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        return F.pad(img, self.pad, mode="constant", value=-1)

class Resize(Transform):
    """
    Upsample
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, size, mode="trilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if len (img.shape) not in [4, 5]:
            raise ValueError("Image should have 4 or 5 dimensions")
        if len(img.shape) == 4:
            img = img[None, ...]
            img = F.interpolate(img, size=self.size, mode=self.mode)
            return img.squeeze(0)
        else:
            return F.interpolate(img, size=self.size, mode=self.mode)