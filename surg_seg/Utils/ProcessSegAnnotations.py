from __future__ import annotations
from dataclasses import dataclass
from typing import List
from surg_seg.Datasets.SegmentationLabelParser import YamlSegMapReader, SegmentationLabelInfo
import click
from pathlib import Path
import numpy as np


@dataclass
class SegAnnotationProcessor:
    mapping_file: Path
    classes_to_keep: List[str]

    def __post_init__(self) -> None:
        self.label_info_reader = YamlSegMapReader(self.mapping_file)

        for c in self.classes_to_keep:
            assert self.label_info_reader.is_class_info_available(
                c
            ), "class not found in mapping file"

    def parse_seg_img(self, seg_img: np.ndarray) -> np.ndarray:
        new_img = np.zeros_like(seg_img)
        for c in self.classes_to_keep:
            class_info = self.label_info_reader.get_class_info(c)

            mask = np.all(seg_img == class_info.rgb, axis=-1)

            new_img[mask] = class_info.rgb

        return new_img


if __name__ == "__main__":
    from PIL import Image

    pass
    input_img = Path("../sim2real/data/pilotdata_01/000001/segmented/000001.png")
    mapping_file = input_img.parent / "meta.yaml"
    output_img = input_img.parent.parent / "segmented_processed" / input_img.name

    output_img.parent.mkdir(exist_ok=True)
    assert mapping_file.exists()
    assert input_img.exists()

    processor = SegAnnotationProcessor(mapping_file=mapping_file, classes_to_keep=["phantom"])

    image = np.array(Image.open(input_img))
    processed_img = processor.parse_seg_img(image)

    Image.fromarray(processed_img).show()
