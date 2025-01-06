import nibabel as nip
import numpy as np
import torch

def norm98(vol):
    q = np.percentile(vol, 98)
    # q is a pixel value, below which 98% of the pixels lie
    vol = vol / q
    vol[vol > 1] = 1
    # return img/self.max_val
    return vol

# csv_path = "data/atlas_skull_stripped/atlas_skull_stripped_test.csv"
csv_path = "data/ixi_skull_stripped/ixi_test.csv"

with open(csv_path, "r") as file:
    data = file.read()
    for line in data.split("\n"):
        line = line.replace(",", "")
        vol = nip.load(line)
        img_np = np.array(vol.get_fdata(), dtype=np.float32)
        img_t = torch.Tensor(img_np.copy())
        img_t = norm98(img_t)
        if torch.isnan(img_t).any():
            print(line)
        