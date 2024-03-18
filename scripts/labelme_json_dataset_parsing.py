from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Dict, List, Tuple
from natsort import natsorted
import os.path as osp
import PIL.Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
from labelme.logger import logger
from labelme import utils
import click
import imgviz

from surg_seg.Datasets.SegmentationLabelParser import (
    SegmentationLabelParser,
    SegmentationLabelInfo,
    YamlSegMapReader,
)

"""
Labelme json file format:
{   "version": "1.0.0", 
    "flags": {}, 
    "shapes": [{
        "label": "needle", 
        "points": [] , 
        "group_id": 2, 
        "description": "", 
        "shape_type": "polygon", 
        "flags": null}], 
    "imagePath": "path", 
    "imageData": imagedata
}

"""


@dataclass
class LabelMeJsonParser:
    """Class to process a single label me json file"""

    json_file: Path
    custom_data_saver: LabelMeImageSaver
    label_parser: SegmentationLabelParser = None

    def __post_init__(self):

        self.data: Dict = json.load(open(self.json_file))

        self.label_name_to_value: Dict[str, int] = {"_background_": 0}
        self.label_names: List[str]
        self.extract_class_names()

        self.img: np.ndarray = self.read_image()
        self.lbl, self.lbl_viz = self.create_labels()

    def read_image(self) -> np.ndarray:
        self.imageData: np.ndarray = self.data.get("imageData")

        # if json does not contain the image data, use the path stored in the json file

        # if not self.imageData:  # if imageData is None
        #     imagePath = os.path.join(os.path.dirname(self.json_file), self.data.get("imageData"))
        #     with open(imagePath, "rb") as f:
        #         self.imageData = f.read()
        #         self.imageData = base64.b64encode(self.imageData).decode("utf-8")

        img = utils.img_b64_to_arr(self.imageData)

        return img

    def extract_class_names(self):
        if self.label_parser is None:
            self.__extract_names_and_create_ids()
        else:
            self.__extract_names_and_id_from_parser()

        self.label_names = [None] * (max(self.label_name_to_value.values()) + 1)
        for name, value in self.label_name_to_value.items():
            self.label_names[value] = name

    def __extract_names_and_create_ids(self):
        for shape in sorted(self.data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in self.label_name_to_value:
                label_value = self.label_name_to_value[label_name]
            else:
                label_value = len(self.label_name_to_value)
                self.label_name_to_value[label_name] = label_value

    def __extract_names_and_id_from_parser(self):
        label_info: SegmentationLabelInfo
        for label_info in self.label_parser.get_classes_info():
            self.label_name_to_value[label_info.name] = label_info.id

    def create_labels(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        The labels in the `shapes` list need to match the keys in the `label_name_to_value` dict
        """

        lbl, _ = utils.shapes_to_label(
            self.img.shape, self.data["shapes"], self.label_name_to_value
        )
        lbl_viz = imgviz.label2rgb(
            lbl, imgviz.asgray(self.img), label_names=self.label_names, loc="rb"
        )

        return lbl, lbl_viz

    def save_without_label_parser(self, out_dir: Path):
        out_dir.mkdir(exist_ok=True, parents=True)

        PIL.Image.fromarray(self.img).save(out_dir / "img.png")
        utils.lblsave(out_dir / "label.png", self.lbl)
        PIL.Image.fromarray(self.lbl_viz).save(osp.join(out_dir, "label_viz.png"))

        with open(osp.join(out_dir, "label_names.txt"), "w") as f:
            for lbl_name in self.label_names:
                f.write(lbl_name + "\n")

        logger.info("Saved to: {}".format(out_dir))

    def save(self):
        if self.custom_data_saver is None:
            raise Exception("Custom data saver was not provided")

        self.custom_data_saver.save_image_and_label(self.img, self.lbl, self.lbl_viz)


class LabelMeImageSaver(ABC):
    def __init__(self, label_parser: SegmentationLabelParser, root_path: Path):
        self.label_parser = label_parser
        self.root_path = root_path

    @abstractmethod
    def save_image_and_label(
        self, img: np.ndarray, lbl: np.ndarray, lbl_viz: np.ndarray
    ) -> None:
        """Abstract method to save an image and its label to a directory.

        Parameters
        ----------
        img : np.ndarray
            RGB raw image
        lbl : np.ndarray
            single channel label img where each pixel contains a class id
        lbl_viz : np.ndarray
            lbl and raw img blended together.
        """
        pass


class MultiDirSaver(LabelMeImageSaver):
    """Save raw images and labels in different folders."""

    def __init__(self, label_parser, root_path, file_prefix: int):
        super().__init__(label_parser, root_path)

        self.file_prefix = file_prefix
        self.raw_dir = self.root_path / "rgb"
        self.labels_dir = self.root_path / "segmented"
        self.blended_dir = self.root_path / "gt_vis"

        self.root_path.mkdir(exist_ok=True, parents=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        self.blended_dir.mkdir(exist_ok=True)

    def save_image_and_label(
        self, img: np.ndarray, lbl: np.ndarray, lbl_viz: np.ndarray
    ):
        name = f"{self.file_prefix:06d}.png"
        PIL.Image.fromarray(img).save(self.raw_dir / (name))

        rgb_lbl = self.label_parser.convert_single_ch_to_rgb(lbl)
        PIL.Image.fromarray(rgb_lbl).save(self.labels_dir / (name))
        PIL.Image.fromarray(lbl_viz).save(self.blended_dir / (name))


class SingleDirSaver(LabelMeImageSaver):
    """
    Save images and labels in same folder
    """

    def save_image_and_label(
        self, img: np.np.ndarray, lbl: np.ndarray, lbl_viz: np.ndarray
    ) -> None:
        self.root_path.mkdir(exist_ok=True, parents=True)

        PIL.Image.fromarray(img).save(self.root_path / "img.png")
        # utils.lblsave(self.root_path / "label.png", lbl)
        rgb_lbl = self.label_parser.convert_single_ch_to_rgb(lbl)
        PIL.Image.fromarray(rgb_lbl).save(self.root_path / "label.png")
        PIL.Image.fromarray(lbl_viz).save(osp.join(self.root_path, "label_viz.png"))

        with open(self.root_path / "label_names.txt", "w") as f:
            for class_info in self.label_parser.get_classes_info():
                f.write(class_info.name + "\n")

        logger.info("Saved to: {}".format(self.root_path))


@click.group()
def mycommands():
    pass


@click.command()
@click.option(
    "--indir", type=Path, required=True, help="Path to the folder with json files"
)
@click.option("--outdir", type=Path, required=True)
@click.option(
    "--labels_yaml_path",
    type=Path,
    help="Path to yaml file with class names and ids",
    required=True,
)
def parse_folder(indir: Path, outdir: Path, labels_yaml_path: Path):

    if not indir.exists():
        raise Exception(
            "Input dir with json_annotations not found in {}".format(indir.resolve())
        )

    if not labels_yaml_path.exists():
        raise Exception("Labels yaml file not found at {}".format(labels_yaml_path))
    else:
        labels_info_reader = YamlSegMapReader(labels_yaml_path)
        labels_parser = SegmentationLabelParser(labels_info_reader)

    json_dir = indir
    json_files = natsorted(json_dir.glob("*.json"))

    for json_file in tqdm(json_files):
        file_prefix = json_file.with_suffix("").name
        file_prefix = int(file_prefix.replace("image_", ""))
        labelme_saver = MultiDirSaver(labels_parser, outdir, file_prefix)
        labelme_json = LabelMeJsonParser(json_file, labelme_saver, labels_parser)
        labelme_json.save()


@click.command()
@click.argument("json_file", type=Path)
@click.option("--outdir", type=Path, required=True)
@click.option(
    "--labels_yaml_path",
    type=Path,
    default=None,
    help="Path to yaml file with class names and ids. If specified labels in yaml file need"
    "to math labels in json file.",
)
def parse_file(json_file: Path, outdir: Path, labels_yaml_path: Path):
    """Parse a single labelme json file and extract raw image and label"""

    folder_name = json_file.with_suffix("").name

    if labels_yaml_path is None:
        labelme_parser = LabelMeJsonParser(json_file, None, None)
        labelme_parser.save_without_label_parser(outdir / folder_name)
    else:
        if not labels_yaml_path.exists():
            raise Exception("Labels yaml file not found at {}".format(labels_yaml_path))

        label_info_reader = YamlSegMapReader(labels_yaml_path)
        label_parser = SegmentationLabelParser(label_info_reader)
        labelme_saver = SingleDirSaver(label_parser, outdir / folder_name)
        labelme_parser = LabelMeJsonParser(json_file, labelme_saver, label_parser)
        labelme_parser.save()


if __name__ == "__main__":
    mycommands.add_command(parse_folder)
    mycommands.add_command(parse_file)

    mycommands()
