import h5py
import torch


class HDF5Dataset(torch.utils.data.Dataset):
    """
    Default hdf5 dataset
    """

    def __init__(
        self,
        h5_filename,
        transform=None,
    ):
        self.h5_filename = h5_filename
        self.transform = transform
        # self.target_transform = target_transform

        with h5py.File(self.h5_filename, "r") as file:
            self.total_num_samples = len(file["label"])

    def __len__(self):
        return self.total_num_samples

    def __getitem__(self, idx):
        if not hasattr(self, "opened_hdf5"):
            self.opened_hdf5 = h5py.File(self.h5_filename, "r")
        image = self.opened_hdf5["image"][idx]
        label = self.opened_hdf5["label"][idx]

        # if self.transform:
        #     if isinstance(self.transform, A.Compose):
        #         transformed = self.transform(image=image)
        #         image = transformed['image']
        #     else:
        #         image = self.transform(image)

        # if image.max() > 1.0:
        #     image = image / 255.0

        return torch.FloatTensor(image), torch.FloatTensor(label)

    def __del__(self):
        if hasattr(self, "opened_hdf5"):
            self.opened_hdf5.close()
