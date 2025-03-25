import pathlib
import torch
from torch.utils.data import Dataset
import os


import SimpleITK as sitk


DATA_PATH = (
    pathlib.Path.home()
    / "Downloads"
    / "Pancreas-CT"
    / "manifest-1599750808610"
    / "Pancreas-CT"
)


def load_DICOM(series_dir: str | pathlib.Path):
    """_summary_

    https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html

    Returns:
        _type_: _description_
    """
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(series_dir)
    if not series_IDs:
        print(f'ERROR: given directory "{series_dir}" does not contain a DICOM series.')
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        series_dir, series_IDs[0]
    )
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # We explicitly configure the reader to load tags, including the
    # private ones.
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()

    return image3D


def resample(image, target_spacing: tuple[float, float, float]):
    # Get current spacing
    original_spacing = image.GetSpacing()

    # Calculate new size
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampled_image = sitk.Resample(
        image1=image,
        size=new_size,
        transform=sitk.Transform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=image.GetOrigin(),
        outputSpacing=target_spacing,
        outputDirection=image.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=image.GetPixelID(),
    )
    return resampled_image


class CTDataset(Dataset):
    def __init__(
        self, img_dir: str | pathlib.Path, transform=None, target_transform=None
    ):
        self._dataset_path = pathlib.Path(img_dir)
        self._data_files = list(self._dataset_path.glob("PANCREAS_????"))
        assert len(self._data_files) > 0, f"No data in the dataset {img_dir}"

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self._data_files)

    def __getitem__(self, idx: int):

        # Specific to the PANCREAS-CT dataset
        data_directory = self._dataset_path / f"PANCREAS_{str(idx+1).rjust(4,'0')}"
        assert data_directory.is_dir(), f"Directory {data_directory} does not exist."
        series_dir = list(list(data_directory.glob("*"))[0].glob("*"))[0]

        image3D = load_DICOM(series_dir=series_dir)

        label = 0  # TODO
        if self.transform:
            image = self.transform(image3D)
        if self.target_transform:
            label = self.target_transform(label)
        return image3D, label


def train():
    # nnUnet variables
    os.environ["nnUNet_raw"] = ""
    os.environ["nnUNet_preprocessed"] = "/home/esadruhn/Downloads/Pancreas-CT/processed"
    os.environ["nnUNet_results"] = (
        "/home/esadruhn/panda_model/data/nnunet_model_weights"
    )

    dataset = CTDataset(
        img_dir=DATA_PATH, transform=lambda x: resample(x, (1.09, 1.09, 3.0))
    )


def main():

    dataset = CTDataset(
        img_dir=DATA_PATH, transform=lambda x: resample(x, (1.09, 1.09, 3.0))
    )
    example_img, example_label = dataset[0]

    unet = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )

    # Train the UNet


if __name__ == "__main__":
    main()
