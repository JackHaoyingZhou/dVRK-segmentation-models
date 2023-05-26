from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import json
import numpy as np
import torch


@dataclass(frozen=True)
class SegmentationLabelInfo:
    """Helper class that relates current rgb color, name and id."""

    id: int
    name: str
    rgb: List[int]


@dataclass
class LabelParser:
    path2mapping: Path
    annotations_type: str
    label_info_reader: LabelInfoReader

    def __post_init__(self) -> None:

        self.__classes_info = self.label_info_reader.classes_info
        self.mask_num = len(self.__classes_info)

        if self.mask_num == 0:
            raise RuntimeError("No classes found. Maybe label_info_reader.read() was not called?")

    def convert_rgb_to_single_channel(self, label_im, color_first=True):
        """Convert an annotations RGB image into a single channel image. The
        label image should have a shape `HWC` where `c==3`. This function
        converts labels that are compatible with Monai.blend function.

        Func to convert indexes taken from
        https://stackoverflow.com/questions/12138339/finding-the-x-y-indexes-of-specific-r-g-b-color-values-from-images-stored-in

        """

        assert label_im.shape[2] == 3, "label in wrong format"

        converted_img = np.zeros((label_im.shape[0], label_im.shape[1]))

        e: SegmentationLabelInfo
        for e in self.__classes_info:
            rgb = e.rgb
            new_color = e.id
            indices = np.where(np.all(label_im == rgb, axis=-1))
            converted_img[indices[0], indices[1]] = new_color

        converted_img = (
            np.expand_dims(converted_img, 0) if color_first else np.expand_dims(converted_img, -1)
        )
        return converted_img

    def convert_rgb_to_onehot(self, mask: np.ndarray):
        """Convert rgb label to one-hot encoding"""

        assert len(mask.shape) == 3, "label not a rgb image"
        assert mask.shape[2] == 3, "label not a rgb image"

        h, w, c = mask.shape

        ## Convert grey-scale label to one-hot encoding
        new_mask = np.zeros((self.mask_num, h, w))

        e: SegmentationLabelInfo
        for e in self.__classes_info:
            rgb = e.rgb
            new_idx = e.id
            new_mask[new_idx, :, :] = np.all(mask == rgb, axis=-1)

        return new_mask

    def convert_onehot_to_single_ch(self, onehot_mask: torch.tensor):
        m_temp = torch.argmax(onehot_mask, axis=0)
        m_temp = torch.unsqueeze(m_temp, 0)
        return m_temp

    # def convert_onehot_to_rgb(self, onehot_mask):

    # new_mask = np.zeros((1, onehot_mask.shape[1], onehot_mask.shape[2]))
    #     e: LabelParserElement
    #     for e in self.conversion_list:

    #         temp = ((m_temp == e.id) * mask_value[idx]).data.numpy()
    #         new_mask += temp
    #     new_mask = np.expand_dims(new_mask, axis=-1)
    #     new_mask = np.concatenate((new_mask, new_mask, new_mask), axis=-1)
    #     new_mask = new_mask.astype(np.int32)
    #     return new_mask


class LabelInfoReader(ABC):
    """Abstract class to read segmentation labels metadata."""

    def __init__(self, mapping_file: Path):
        self.mapping_file = mapping_file
        self.classes_info: List[SegmentationLabelInfo] = []

    @abstractmethod
    def read(self):
        """Read file and construct the classes_info list."""
        pass


class AmbfMultiClassesInfoReader(LabelInfoReader):
    """Read the mapping file for ambf multi-class segmentation."""

    def __init__(self, mapping_file: Path, annotations_type: str):
        """
        Read segmentation labels mapping files

        Parameters
        ----------
        mapping_file : Path
        annotation_type : str
            Either [2colors, 4colors, or 5colors]
        """

        super().__init__(mapping_file)
        self.annotations_type = annotations_type

    def read(self):

        with open(self.mapping_file, "r") as f:
            mapper = json.load(f)

        if self.annotations_type in mapper:
            mask = mapper[self.annotations_type]
        else:
            raise RuntimeWarning(
                f"annotations type {self.annotations_type} not found in {self.path2mapping}"
            )

        self.classes_info = [
            SegmentationLabelInfo(idx, key, value) for idx, (key, value) in enumerate(mask.items())
        ]


class DvrkClassesInfoReader(LabelInfoReader):
    pass
