import argparse
import base64
import json
import os
from natsort import natsorted
import os.path as osp

import PIL.Image
import yaml
from pathlib import Path
from labelme.logger import logger
from labelme import utils
import click
import imgviz


def process_single_file(json_file: Path, out_dir: Path):

    out_dir = out_dir / json_file.name
    out_dir.mkdir(exist_ok=True)

    data = json.load(open(json_file))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    lbl_viz = imgviz.label2rgb(lbl, imgviz.asgray(img), label_names=label_names, loc="rb")

    PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))
    utils.lblsave(osp.join(out_dir, "label.png"), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))

    with open(osp.join(out_dir, "label_names.txt"), "w") as f:
        for lbl_name in label_names:
            f.write(lbl_name + "\n")

    logger.info("Saved to: {}".format(out_dir))


@click.command()
@click.argument("indir", type=Path)
@click.option("--outdir", default=None, type=Path, help="Output image directory.")
def parse_folder(indir: Path, outdir: Path):
    """Parse a folder with labelme json files into a dataset of images and labels."""

    json_files = natsorted(indir.glob("*.json"))
    if not outdir.exists():
        outdir = indir / "annotations"
        outdir.mkdir(exist_ok=True)

    outdir = outdir.resolve()
    print(outdir)

    for file in json_files:
        print(file)
        process_single_file(file, outdir)


if __name__ == "__main__":
    parse_folder()

    # print(["Hello"])
    # {
    #     print("Hello")
    #     print("Hello")
    # }
