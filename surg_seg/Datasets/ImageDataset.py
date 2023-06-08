import json
from pathlib import Path
from typing import List
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import re
import natsort

from monai.visualize.utils import blend_images
from dataclasses import InitVar, dataclass, field
from surg_seg.Datasets.SegmentationLabelParser import LabelInfoReader, SegmentationLabelParser


class ImageTransforms:

    img_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalize
        ]
    )
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    inv_transforms = T.Compose(
        [
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )


@dataclass
class ImageSegmentationDataset(Dataset):
    images_list: List[Path]
    labels_list: List[Path]
    label_parser: SegmentationLabelParser

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx, transform=True):
        if isinstance(idx, slice):
            RuntimeError("Slices are not supported")

        image = np.array(Image.open(self.images_list[idx]))
        annotation = np.array(Image.open(self.labels_list[idx]))

        if transform:
            image = ImageTransforms.img_transforms(image)
            annotation = self.label_parser.convert_rgb_to_onehot(annotation)
            annotation = torch.tensor(annotation)

        return {"image": image, "label": annotation}


class Ambf5RecDataReader:
    def __init__(self, root_dirs: List[Path], annotation_type: str):
        """Image dataset

        Parameters
        ----------
        root_dir : Path
        annotation_type : str
            Either [2colors, 4colors, or 5colors]
        """

        if not isinstance(root_dirs, list):
            root_dirs = [root_dirs]

        self.image_folder_list = []
        self.images_list = []
        self.labels_list = []
        for root_dir in root_dirs:
            single_folder = SingleFolderReader(root_dir, annotation_type)
            self.image_folder_list.append(single_folder)
            self.images_list += single_folder.images_path_list
            self.labels_list += single_folder.label_path_list

    def __len__(self):
        return len(self.images_list)


@dataclass
class SingleFolderReader:
    """
    Read a single folder of data from the Ambf5Rec dataset
    """

    root_dir: Path
    annotation_type: InitVar[str]
    annotation_path: Path = field(init=False)
    image_path_list: List[Path] = field(init=False)
    label_path_list: List[Path] = field(init=False)
    image_id_list: List[int] = field(init=False)
    # Auxiliary variables used to identify duplicated ids in image folder
    flag_list: List[int] = field(init=False)

    def __post_init__(self, annotation_type):

        self.annotation_dir = self.__get_annotation_dir(annotation_type)

        self.images_path_list = natsort.natsorted(list((self.root_dir / "raw").glob("*.png")))
        self.flag_list = np.zeros(len(self.images_path_list))
        self.images_id_list = self.compute_id_list()

        self.label_path_list = [self.annotation_dir / img.name for img in self.images_path_list]

    def compute_id_list(self):
        ids = []
        img_name: Path
        for img_name in self.images_path_list:
            id_match = self.__extract_id(img_name.name)
            self.__check_and_mark_id(id_match)
            ids.append(id_match)
        return ids

    def __get_annotation_dir(self, annotation_type):
        valid_options = ["2colors", "4colors", "5colors"]
        if annotation_type not in valid_options:
            raise RuntimeError(
                f"{annotation_type} is not a valid annotation.\n Valid annotations are {valid_options}"
            )
        return self.root_dir / ("annotation" + annotation_type)

    def __extract_id(self, img_name: str) -> int:
        """Extract id from image name"""
        id_match = re.findall("[0-9]{6}", img_name)

        if len(id_match) == 0:
            raise RuntimeError(f"Image {img_name} not formatted correctly")

        id_match = int(id_match[0])
        return id_match

    def __check_and_mark_id(self, id_match):
        """Check that there are no duplicated id"""
        if self.flag_list[id_match]:
            raise RuntimeError(f"Id {id_match} is duplicated")

        self.flag_list[id_match] = 1


def display_transformed_images(idx: int, ds: ImageSegmentationDataset):

    data = ds[idx]  # Get transformed images
    # print(f"one-hot shape: {data['label'].shape}")

    single_ch_annotation = ds.label_parser.convert_onehot_to_single_ch(data["label"])
    single_ch_annotation = np.array(single_ch_annotation)
    # single_ch_annotation = ds.label_parser.convert_rgb_to_single_channel(raw_label)
    raw_image = np.array(ImageTransforms.inv_transforms(data["image"]))
    raw_label = ds.__getitem__(idx, transform=False)["label"]

    blended = blend_images(
        raw_image,
        single_ch_annotation,
        cmap="viridis",
        alpha=0.8,
    )

    display_images(raw_image.transpose(1, 2, 0), raw_label, blended.transpose(1, 2, 0))


def display_untransformed_images(idx: int, ds: ImageSegmentationDataset):
    data = ds.__getitem__(idx, transform=False)  # Get raw images

    raw_image = data["image"]
    raw_label = data["label"]
    onehot = ds.label_parser.convert_rgb_to_onehot(raw_label)

    # fake_annotation = np.zeros_like(np.array(data["image"]))
    # fake_annotation[:40, :40] = [1, 1, 1]
    # fake_annotation[40:80, 40:80] = [2, 2, 2]
    # fake_annotation[80:120, 80:120] = [3, 3, 3]

    single_ch_label = ds.label_parser.convert_rgb_to_single_channel(raw_label)
    blended = blend_images(
        raw_image.transpose(2, 0, 1),
        single_ch_label,
        cmap="viridis",
        alpha=0.8,
    )

    display_images(raw_image, raw_label, blended.transpose(1, 2, 0))


def display_images(img, label, blended):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(label)
    ax[2].imshow(blended)
    [a.set_axis_off() for a in ax.squeeze()]
    fig.set_tight_layout(True)
    plt.show()


if __name__ == "__main__":
    root = Path("/home/juan1995/research_juan/accelnet_grant/data")
    data_dirs = [root / "rec01", root / "rec02", root / "rec03", root / "rec04", root / "rec05"]

    ds = ImageSegmentationDataset(data_dirs, "5colors")

    print(f"length of dataset: {len(ds)}")

    display_untransformed_images(100, ds)
    display_transformed_images(100, ds)
    display_transformed_images(230, ds)
    display_transformed_images(330, ds)
    display_transformed_images(430, ds)
    display_transformed_images(530, ds)
