from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from natsort import natsorted
import numpy as np
from typing import List, Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm
from monai.visualize.utils import blend_images
import torch
from monai.bundle import ConfigParser
from monai.data import ThreadDataLoader
from monai.networks.nets import FlexibleUNet
from PIL import Image

from surg_seg.Datasets.SegmentationLabelParser import (
    SegmentationLabelParser,
    YamlSegMapReader,
)
from surg_seg.Datasets.ImageDataset import (
    ImageDirParser,
    ImageSegmentationDataset,
)
from surg_seg.ImageTransforms.ImageTransforms import ImageTransforms
from surg_seg.Metrics.MetricsUtils import AggregatedMetricTable, IOUStats
from surg_seg.Networks.Models import FlexibleUnet1InferencePipe, create_FlexibleUnet
from surg_seg.Trainers.Trainer import ModelTrainer

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from surg_seg.HydraConfig.SegConfig import SegmentationConfig

import sys
from train_phantom_segmentation import load_dataset, create_model, DatasetContainer

print(sys.path)

cs = ConfigStore.instance()
cs.store(name="base_config", node=SegmentationConfig)

def save_test_predictions(
    config: SegmentationConfig,
    dataset_container: DatasetContainer,
    model_pipe: FlexibleUnet1InferencePipe,
):
    predictions_dir = Path(config.path_config.predictions_path).resolve()
    predictions_dir.mkdir(exist_ok=True)
    ds = dataset_container.ds_test

    # fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    # fig.set_tight_layout(True)
    # fig.subplots_adjust(hspace=0, wspace=0)
    # # for i, ax in enumerate(axes.flat):

    count = 0
    total_num = len(ds)

    for i in range(len(ds)):
        # pair = next(iter(dl))
        pair = ds.get_sample(i, transform=False)
        im = pair["image"]
        im_path = Path(pair["path"])
        pred_name = im_path.parent.parent.name + "_" + im_path.stem + "_pred.png"

        input_tensor, inferred_single_ch = model_pipe.infer(im)
        inferred_single_ch = inferred_single_ch.detach().cpu()
        input_tensor = input_tensor.detach().cpu()[0]

        blended = blend_images(input_tensor, inferred_single_ch, cmap="viridis", alpha=0.8).numpy()
        blended = (np.transpose(blended, (1, 2, 0)) * 254).astype(np.uint8)
        Image.fromarray(blended).save(predictions_dir / pred_name)

        # im = ImageTransforms.inv_transforms(im)
        # lb = label_parser.convert_onehot_to_single_ch(lb)
        # blended = blend_images(im, lb, cmap="viridis", alpha=0.7)
        # blended = blended.numpy().transpose(1, 2, 0)
        # blended = (blended * 255).astype(np.uint8)
        count += 1
        sys.stdout.write(f"\r Run Progress: {count} / {total_num}")
        sys.stdout.flush()

        # ax.imshow(blended)
        # ax.axis("off")
    # plt.show()


def calculate_metrics_on_valid(
    config: SegmentationConfig,
    data_container: DatasetContainer,
    model_pipe: FlexibleUnet1InferencePipe,
):
    device = config.test_config.device

    label_parser = data_container.label_parser
    dl = data_container.dl_test

    iou_stats = IOUStats(label_parser)
    for batch in tqdm(dl, desc="Calculating metrics"):
        img = batch["image"].to(device)
        label = batch["label"]
        img_paths = ["empty"] * label.shape[0]

        prediction = model_pipe.model(img).detach().cpu()
        onehot_prediction = ImageTransforms.predictions_transforms(prediction)
        iou_stats.calculate_metrics_from_batch(onehot_prediction, label, img_paths)

        img = img.detach().cpu()[0]
        # img = ImageTransforms.inv_transforms(img).type(torch.uint8)[0].numpy()
        single_ch_prediction = onehot_prediction[0].argmax(dim=0, keepdim=True)
        blended = blend_images(img, single_ch_prediction, cmap="viridis", alpha=0.8).numpy()
        blended = (np.transpose(blended, (1, 2, 0)) * 254).astype(np.uint8)

        ### show image
        # fig, ax = plt.subplots(1, 1)
        # plt.imshow(blended, cmap="")
        # ax.imshow(blended)
        # ax.imshow(np.transpose(img, (1, 2, 0)))
        # x, y = np.where(single_ch_prediction[0].numpy() == 1)
        # if len(x) == 0:
        #     print(f"{x[0]}, {y[0]}, len: {len(x)}")
        # else:
        #     print("no needle")
        # # plt.show()
        # print("hello")

    iou_stats.calculate_aggregated_stats()
    table = AggregatedMetricTable(iou_stats)
    table.fill_table()
    table.print_table()

def test_model(
    config: SegmentationConfig, dataset_container: DatasetContainer, model: FlexibleUNet
):
    # path2weights = Path(config.path_config.trained_weights_file_path)
    # path2weights /= config.path_config.trained_weights_file_name

    path2weights = Path(config.path_config.trained_weights_path)

    label_parser = dataset_container.label_parser
    ds = dataset_container.ds_test

    model_pipe = FlexibleUnet1InferencePipe(
        path2weights, config.test_config.device, out_channels=label_parser.mask_num, model=model
    )
    model_pipe.model.eval()
    model_pipe.upload_weights()

    if config.actions.save_test_predictions:
        save_test_predictions(config, dataset_container, model_pipe)

    if config.actions.calculate_metrics:
        calculate_metrics_on_valid(config, dataset_container, model_pipe)


@hydra.main(version_base=None, config_path="../../config/phantom_seg", config_name="phantom_seg_jack_newtool")
# config_name="phantom_seg_jack_oldtool"
# config_name="phantom_seg_jack_mixtool"
def main(cfg: SegmentationConfig):
    print(OmegaConf.to_yaml(cfg))

    dataset_container = load_dataset(cfg)
    model = create_model(cfg, dataset_container)

    test_model(cfg, dataset_container, model)


if __name__ == "__main__":
    main()
