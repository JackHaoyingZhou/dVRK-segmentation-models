from typing import Dict, List
import torch
import pytest
from pathlib import Path
from PIL import Image
from natsort import natsorted
import numpy as np
from monai.metrics.meaniou import compute_iou, compute_meaniou
from surg_seg.Datasets.SegmentationLabelParser import YamlSegMapReader, SegmentationLabelParser

data_path = Path(__file__).parent / "./Data/"


def img2tensor(paths: List[Path]):
    list_of_tensors = []
    for path in paths:
        img = np.array(Image.open(path))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        list_of_tensors.append(torch.tensor(img))

    list_of_tensors = np.concatenate(list_of_tensors, axis=0)
    return torch.tensor(list_of_tensors)


def label2tensor(paths: List[Path], label_parser: SegmentationLabelParser):
    list_of_tensors = []
    for path in paths:
        label = np.array(Image.open(path))
        label = label_parser.convert_rgb_to_onehot(label)
        label = np.expand_dims(label, axis=0)
        list_of_tensors.append(torch.tensor(label))

    list_of_tensors = np.concatenate(list_of_tensors, axis=0)
    return torch.tensor(list_of_tensors)


@pytest.fixture
def label_parser() -> SegmentationLabelParser:
    seg_map_reader = YamlSegMapReader(data_path / "dataset_config.yaml")
    label_parser = SegmentationLabelParser(seg_map_reader)

    return label_parser


@pytest.fixture
def sample_data(label_parser: SegmentationLabelParser):

    raw_img = data_path / "raw"
    raw_img = natsorted(list(raw_img.glob("*.png")))
    raw_img = img2tensor(raw_img)

    label = data_path / "label"
    label = natsorted(list(label.glob("*.png")))
    label = label2tensor(label, label_parser)

    return {"raw": raw_img, "label": label}


def test_sample_data_path():
    assert data_path.exists(), f"{data_path} does not exist"


def test_raw_img_dimensions(sample_data):
    raw_img = sample_data["raw"]
    assert raw_img.shape == (3, 3, 1024, 1280), "Raw image shape is {}".format(raw_img.shape)


def test_label_dimensions(sample_data):
    label = sample_data["label"]
    assert label.shape == (3, 3, 1024, 1280), "Label shape is {}".format(label.shape)


def test_label_parser(sample_data, label_parser: SegmentationLabelParser):
    assert label_parser.mask_num == 3, "Mask num is {}".format(label_parser.mask_num)


def test_single_label_iou_dimensions(sample_data):
    label = sample_data["label"]

    one_hotlabel = torch.unsqueeze(label[0], 0)
    iou_tensor = compute_iou(one_hotlabel, one_hotlabel, include_background=False)
    assert iou_tensor.shape == (1, 2), "IOU tensor shape is {}".format(iou_tensor.shape)


def test_single_label_iou_value(sample_data: Dict[str, torch.Tensor]):
    label = sample_data["label"]
    one_hotlabel = torch.unsqueeze(label[0], 0)
    iou_tensor = compute_iou(one_hotlabel, one_hotlabel, include_background=False)

    answer = torch.tensor([[1.0, 1.0]])
    assert torch.allclose(iou_tensor, answer), "IOU tensor value is {}".format(iou_tensor)


def test_batch_label_iou_value(sample_data: Dict[str, torch.Tensor]):
    label = sample_data["label"]
    iou_tensor = compute_iou(label, label, include_background=False)

    answer = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    assert torch.allclose(iou_tensor, answer), "IOU tensor value is {}".format(iou_tensor)
