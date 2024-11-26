import glob

from transforms.preprocessing import *
from data.loaders.base_loader import BaseLoader

from dl_utils import get_data_from_csv
from dl_utils.mask_utils import dilate_mask


class AtlasLoader(BaseLoader):
    def __init__(
        self,
        data_dir,
        file_type="",
        label_dir=None,
        mask_dir=None,
        target_size=(256, 256),
        test=False,
        dilation_kernel=1,
    ):
        self.dilation_kernel = dilation_kernel
        self.mask_dir = mask_dir
        if mask_dir is not None:
            if "csv" in mask_dir[0]:
                self.mask_files = get_data_from_csv(mask_dir)
            else:
                self.mask_files = [
                    glob.glob(mask_dir_i + file_type) for mask_dir_i in mask_dir
                ]
        super(AtlasLoader, self).__init__(
            data_dir, file_type, label_dir, mask_dir, target_size, test
        )

    def get_label(self, idx):
        patho_mask = 0
        brain_mask = 0

        if self.label_dir is not None:
            patho_mask = self.seg_t(self.label_files[idx])
        if self.mask_dir is not None:
            brain_mask = self.seg_t(self.mask_files[idx])
        return (patho_mask, brain_mask)

    def get_dilated_mask(self, idx):
        dilated_patho_mask = 0
        if self.label_dir is not None and self.mask_dir is not None:
            (patho_mask, brain_mask) = self.get_label(idx)

            dilated_patho_mask = dilate_mask(patho_mask, kernel=self.dilation_kernel)
            dilated_patho_mask[brain_mask == 0] = 0

        return dilated_patho_mask

    def get_filename(self, idx):
        return self.files[idx]

    def get_mask_filename(self, idx):
        if self.label_dir is not None:
            return self.label_files[idx]
        else:
            return ""

    def __getitem__(self, idx):
        return (
            self.im_t(self.files[idx]),
            *self.get_label(idx),
            self.get_dilated_mask(idx),
            self.get_filename(idx),  # filename
            self.get_mask_filename(idx),  # mask_filename
        )


class AtlasLoaderPalette(AtlasLoader):
    def __init__(self, *args, **kwargs):
        super(AtlasLoaderPalette, self).__init__(*args, **kwargs)

    def get_palette_mask(self, idx):
        palette_mask = 0
        if self.label_dir is not None:
            patho_mask, _ = self.get_label(idx)
            image = self.im_t(self.files[idx])
            palette_mask = (
                patho_mask * torch.randn_like(patho_mask) + (1 - patho_mask) * image
            )
        return palette_mask

    def __getitem__(self, idx):
        return (
            self.im_t(self.files[idx]),
            *self.get_label(idx),
            self.get_dilated_mask(idx),
            self.get_palette_mask(idx),
            self.get_filename(idx),  # filename
            self.get_mask_filename(idx),  # mask_filename
        )
