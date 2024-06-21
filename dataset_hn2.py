import os
import nibabel as nib
import numpy as np
import random
from torch.utils.data import Dataset


def normalize(img):
    norm_img = np.clip(img, -1024, 1024) / 1024
    return norm_img


class HNDataset(Dataset):
    def __init__(self, img_dir, shift, transform=None):
        self.image_dir = img_dir
        self.transform = transform
        self.shift = shift
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def get_locations(self, gt, x_lim, y_lim, z_lim):
        random_loc = random.choice(np.argwhere(gt == 1))
        x = random_loc[0] + np.random.randint(-x_lim, x_lim)
        y = random_loc[1] + np.random.randint(-y_lim, y_lim)
        z = random_loc[2] + np.random.randint(-z_lim, z_lim)
        return x, y, z

    def augment_locations(self, patch_shape, dims, x, y, z):
        x_min = x - (patch_shape[0] // 2)
        x_max = x + (patch_shape[0] // 2)
        y_min = y - (patch_shape[1] // 2)
        y_max = y + (patch_shape[1] // 2)
        z_min = z - (patch_shape[2] // 2)
        z_max = z + (patch_shape[2] // 2)
        sampling_loc = [x_min, x_max, y_min, y_max, z_min, z_max]

        if sampling_loc[0] < 0:
            sampling_loc[0] = 0
            sampling_loc[1] = sampling_loc[0] + patch_shape[0]

        if sampling_loc[1] > dims[0]:
            sampling_loc[1] = dims[0]
            sampling_loc[0] = sampling_loc[1] - patch_shape[0]

        if sampling_loc[2] < 0:
            sampling_loc[2] = 0
            sampling_loc[3] = sampling_loc[2] + patch_shape[1]

        if sampling_loc[3] > dims[1]:
            sampling_loc[3] = dims[1]
            sampling_loc[2] = sampling_loc[3] - patch_shape[1]

        if sampling_loc[4] < 0:
            sampling_loc[4] = 0
            sampling_loc[5] = sampling_loc[4] + patch_shape[2]

        if sampling_loc[5] > dims[2]:
            sampling_loc[5] = dims[2]
            sampling_loc[4] = sampling_loc[5] - patch_shape[2]
        return sampling_loc

    def sample_patches(self, ct, gt, sampling_loc):
        ct_patch = ct[sampling_loc[0]:sampling_loc[1],
                      sampling_loc[2]:sampling_loc[3],
                      sampling_loc[4]:sampling_loc[5]]
        gt_patch = gt[sampling_loc[0]:sampling_loc[1],
                      sampling_loc[2]:sampling_loc[3],
                      sampling_loc[4]:sampling_loc[5]]
        return ct_patch, gt_patch

    def __getitem__(self, index):
        patient = os.path.join(self.image_dir, self.images[index])

        ct_path = os.path.join(patient, "CT/CT.nii.gz")
        gt_path = os.path.join(patient, "GT/GT.nii.gz")

        ct = nib.load(ct_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()

        gt[gt > 0] = 1

        ct = normalize(ct)
        dims = np.shape(ct)
        patch_shape = [144, 144, 144]

        x, y, z = self.get_locations(gt,
                                     x_lim=self.shift,
                                     y_lim=self.shift,
                                     z_lim=self.shift)

        sampling_loc = self.augment_locations(patch_shape, dims, x, y, z)
        ct_patch, gt_patch = self.sample_patches(ct, gt, sampling_loc)

        sample = dict()
        sample['input'] = ct_patch
        sample['target'] = gt_patch
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class HNDatasetVal(Dataset):
    def __init__(self, img_dir, transform=None):
        self.image_dir = img_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def augment_locations(self, patch_shape, dims, x, y, z):
        x_min = x - (patch_shape[0] // 2)
        x_max = x + (patch_shape[0] // 2)
        y_min = y - (patch_shape[1] // 2)
        y_max = y + (patch_shape[1] // 2)
        z_min = z - (patch_shape[2] // 2)
        z_max = z + (patch_shape[2] // 2)
        sampling_loc = [x_min, x_max, y_min, y_max, z_min, z_max]

        if sampling_loc[0] < 0:
            sampling_loc[0] = 0
            sampling_loc[1] = sampling_loc[0] + patch_shape[0]

        if sampling_loc[1] > dims[0]:
            sampling_loc[1] = dims[0]
            sampling_loc[0] = sampling_loc[1] - patch_shape[0]

        if sampling_loc[2] < 0:
            sampling_loc[2] = 0
            sampling_loc[3] = sampling_loc[2] + patch_shape[1]

        if sampling_loc[3] > dims[1]:
            sampling_loc[3] = dims[1]
            sampling_loc[2] = sampling_loc[3] - patch_shape[1]

        if sampling_loc[4] < 0:
            sampling_loc[4] = 0
            sampling_loc[5] = sampling_loc[4] + patch_shape[2]

        if sampling_loc[5] > dims[2]:
            sampling_loc[5] = dims[2]
            sampling_loc[4] = sampling_loc[5] - patch_shape[2]
        return sampling_loc

    def sample_patches(self, ct, gt, sampling_loc):
        ct_patch = ct[sampling_loc[0]:sampling_loc[1],
                      sampling_loc[2]:sampling_loc[3],
                      sampling_loc[4]:sampling_loc[5]]
        gt_patch = gt[sampling_loc[0]:sampling_loc[1],
                      sampling_loc[2]:sampling_loc[3],
                      sampling_loc[4]:sampling_loc[5]]
        return ct_patch, gt_patch

    def __getitem__(self, index):
        patient = os.path.join(self.image_dir, self.images[index])

        ct_path = os.path.join(patient, "CT/CT.nii.gz")
        gt_path = os.path.join(patient, "GT/GT.nii.gz")

        ct = nib.load(ct_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
        gt[gt > 0] = 1

        ct = normalize(ct)
        dims = np.shape(ct)
        patch_shape = [144, 144, 144]
        pos = np.argwhere((gt == 1))
        x, y, z = random.choice(pos)
        sampling_loc = self.augment_locations(patch_shape, dims, x, y, z)
        ct_patch, gt_patch = self.sample_patches(ct, gt, sampling_loc)

        sample = dict()
        sample['input'] = ct_patch
        sample['target'] = gt_patch

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
