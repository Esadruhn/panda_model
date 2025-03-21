import pathlib
import torch
from torch.utils.data import Dataset
import nibabel as nib

DATA_PATH = (
    pathlib.Path.home()
    / "Downloads"
    / "Pancreas-CT"
    / "manifest-1599750808610"
    / "Pancreas-CT"
)


class CTDataset(Dataset):
    def __init__(
        self, img_dir: str | pathlib.Path, transform=None, target_transform=None
    ):
        self._dataset_path = pathlib.Path(img_dir)
        self._data_files = list(self._dataset_path.glob("*.nii"))
        assert len(self._data_files) > 0, f"No data in the dataset {img_dir}"

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self._data_files)

    def __getitem__(self, idx: int):
        img_path = self._data_files[idx]
        image = nib.load(img_path)
        label = 0  # TODO
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def main():
    dataset = CTDataset(img_dir=DATA_PATH)
    example_img, example_label = dataset[0]
    print(example_img.header)
    print(example_img.affine)
    sx, sy, sz = example_img.header.get_zooms()
    print(sx)
    print(sy)
    print(sz)


if __name__ == "__main__":
    main()
