from typing import List
from natsort import natsorted
import numpy as np
from pathlib import Path
import click
from tqdm import tqdm
from surg_seg.Datasets.ImageDataset import ImageDirParser, ImageSegmentationDataset
import cv2
from monai.visualize.utils import blend_images
from surg_seg.Datasets.SegmentationLabelParser import (
    SegmentationLabelParser,
    YamlSegMapReader,
)
from surg_seg.ImageTransforms.ImageTransforms import ImageTransforms


class BopImageDirParser(ImageDirParser):
    def __init__(self, root_dirs: List[Path]):
        super().__init__(root_dirs)

        self.parse_image_dir(root_dirs[0])

    def parse_image_dir(self, root_dir: Path):
        self.images_list = natsorted(list(root_dir.glob("rgb/*.png")))
        self.labels_list = natsorted(list(root_dir.glob("segmented/*.png")))


def create_label_parser(mapping_file: Path) -> SegmentationLabelParser:
    assert mapping_file.exists(), f"Mapping file {mapping_file} does not exist"
    label_info_reader = YamlSegMapReader(mapping_file)
    label_parser = SegmentationLabelParser(label_info_reader)

    return label_parser


def load_dataset(dataset_path: Path, seg_map_path: Path):

    label_parser = create_label_parser(seg_map_path)
    data_reader = BopImageDirParser([dataset_path])
    ds = ImageSegmentationDataset(label_parser, data_reader, color_transforms=None)

    return ds, label_parser


@click.command()
@click.option(
    "--dataset_path",
    type=click.Path(exists=True, path_type=Path, resolve_path=True),
    help="Path to the dataset directory. It should look something like: /path/to/dataset/0001/",
)
@click.option(
    "--output_path",
    type=click.Path(path_type=Path, resolve_path=True),
    default=None,
    help="Path to the output directory. Default same as dataset_path",
)
@click.option(
    "--visualize",
    is_flag=True,
    default=False,
    help="If added show images in cv2 window",
)
@click.option(
    "--seg_map_path", type=click.Path(exists=True, path_type=Path), default=None
)
def main(dataset_path: Path, output_path: Path, visualize: bool, seg_map_path) -> None:
    """
    Create visualization images for ground truth segmentation annotations.
    """
    if output_path is None:
        output_path = dataset_path / "gt_seg_visualization"
    else:
        output_path = output_path / "gt_seg_visualization"
    output_path.mkdir(exist_ok=True, parents=True)

    if seg_map_path is None:
        seg_map_path = dataset_path / "../segmap.yaml"

    ds, label_parser = load_dataset(dataset_path, seg_map_path)

    for count, sample in tqdm(enumerate(ds), total=len(ds)):
        img = sample["image"]
        lb = sample["label"]
        # img = img.cpu().numpy().transpose(1, 2, 0)
        img = ImageTransforms.inv_transforms(img)
        lb = label_parser.convert_onehot_to_single_ch(lb)
        blended = blend_images(img, lb, cmap="viridis", alpha=0.7)
        blended = blended.numpy().transpose(1, 2, 0)
        blended = (blended * 255).astype(np.uint8)
        blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(output_path / f"{count:06d}.png"), blended)

        if visualize:
            cv2.imshow("img", blended)

            k = cv2.waitKey(30)

            if k == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
