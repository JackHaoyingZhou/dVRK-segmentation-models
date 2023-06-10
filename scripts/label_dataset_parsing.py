import argparse
import base64
import json
import os
from typing import Dict, List, Tuple
from natsort import natsorted
import os.path as osp

import PIL.Image
import numpy as np
import yaml
from pathlib import Path
from labelme.logger import logger
from labelme import utils
import click
import imgviz

"""
Label me json file format:
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


class LabelMeParser:
    """Class to process a single label me json file"""

    def __init__(self, json_file: Path):

        self.json_file: Path = json_file
        self.data: Dict = json.load(open(json_file))

        self.img: np.ndarray = self.read_image()
        self.lbl, self.lbl_viz, self.label_names = self.create_labels()

    def read_image(self) -> np.ndarray:
        self.imageData: np.ndarray = self.data.get("imageData")

        if not self.imageData:  # if imageData is None
            imagePath = os.path.join(os.path.dirname(self.json_file), self.data.get("imageData"))
            with open(imagePath, "rb") as f:
                self.imageData = f.read()
                self.imageData = base64.b64encode(self.imageData).decode("utf-8")

        img = utils.img_b64_to_arr(self.imageData)

        return img

    def create_labels(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        # parse label names
        label_name_to_value: Dict[str, int] = {"_background_": 0}
        for shape in sorted(self.data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl, _ = utils.shapes_to_label(self.img.shape, self.data["shapes"], label_name_to_value)
        lbl_viz = imgviz.label2rgb(lbl, imgviz.asgray(self.img), label_names=label_names, loc="rb")

        return lbl, lbl_viz, label_names

    def save_images_and_labels(self, out_dir: Path):
        out_dir.mkdir(exist_ok=True)

        PIL.Image.fromarray(self.img).save(out_dir / "img.png")
        utils.lblsave(out_dir / "label.png", self.lbl)
        PIL.Image.fromarray(self.lbl_viz).save(osp.join(out_dir, "label_viz.png"))

        with open(osp.join(out_dir, "label_names.txt"), "w") as f:
            for lbl_name in self.label_names:
                f.write(lbl_name + "\n")

        logger.info("Saved to: {}".format(out_dir))


def __parse_single_file(json_file: Path, out_dir: Path):
    """Parse a single labelme json file into a dataset of images and labels."""
    parser = LabelMeParser(json_file)
    folder_name = json_file.with_suffix("").name
    parser.save_images_and_labels(out_dir / folder_name)


@click.group()
def mycommands():
    pass


@click.command()
@click.argument("indir", type=Path)
def parse_folder(indir: Path):
    """Parse a dataset with multiple labelme json files.
    Indir should contain a directory json_annotations with the json files.
    """

    if not (indir / "json_annotations").exists():
        raise Exception("Directory json_annotations not found in {}".format(indir))

    json_dir = indir / "json_annotations"
    json_files = natsorted(json_dir.glob("*.json"))

    raw_dir = indir / "raw"
    labels_dir = indir / "labels"
    blended_dir = indir / "blended"

    raw_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    blended_dir.mkdir(exist_ok=True)

    for json_file in json_files:
        file_prefix = json_file.with_suffix("").name

        parser = LabelMeParser(json_file)

        PIL.Image.fromarray(parser.img).save(raw_dir / (file_prefix + "_img.png"))
        utils.lblsave(labels_dir / (file_prefix + "_label.png"), parser.lbl)
        PIL.Image.fromarray(parser.lbl_viz).save(
            osp.join(blended_dir, (file_prefix + "_label_viz.png"))
        )


@click.command()
@click.argument("json_file", type=Path)
@click.argument("out_dir", type=Path)
def parse_single_json_file(json_file: Path, out_dir: Path):
    """
    Parse a single labelme json file.
    """
    __parse_single_file(json_file, out_dir)


if __name__ == "__main__":
    mycommands.add_command(parse_folder)
    mycommands.add_command(parse_single_json_file)

    mycommands()
