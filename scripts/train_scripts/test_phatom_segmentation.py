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


def test_model(config: SegmentationConfig, dataset_container: DatasetContainer, model):
    ds = dataset_container.ds_test
    label_parser = dataset_container.label_parser
    path2weights = Path(config.path_config.trained_weights_path)

    model_pipe = FlexibleUnet1InferencePipe(
        path2weights, config.test_config.device, out_channels=label_parser.mask_num, model=model
    )
    model_pipe.upload_weights()

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    fig.set_tight_layout(True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i, ax in enumerate(axes.flat):
        # pair = next(iter(dl))
        pair = ds.__getitem__(i, transform=False)
        im = pair["image"]
        lb = pair["label"]
        print(im.shape)

        # prediction 1
        input_tensor, inferred_single_ch = model_pipe.infer(im)
        inferred_single_ch = inferred_single_ch.detach().cpu()
        input_tensor = input_tensor.detach().cpu()[0]
        blended1 = blend_images(input_tensor, inferred_single_ch, cmap="viridis", alpha=0.8).numpy()
        blended1 = (np.transpose(blended1, (1, 2, 0)) * 254).astype(np.uint8)

        # prediction 2
        input_tensor2 = ImageTransforms.img_transforms(im).to(config.test_config.device)
        input_tensor2 = torch.unsqueeze(input_tensor2, 0)
        prediction = model_pipe.model(input_tensor2).detach().cpu()
        onehot_prediction = ImageTransforms.predictions_transforms(prediction)
        single_ch_prediction = onehot_prediction[0].argmax(dim=0, keepdim=True)
        input_tensor2 = input_tensor2.detach().cpu()[0]

        blended2 = blend_images(
            input_tensor2, single_ch_prediction, cmap="viridis", alpha=0.8
        ).numpy()
        blended2 = (np.transpose(blended2, (1, 2, 0)) * 254).astype(np.uint8)

        # im = ImageTransforms.inv_transforms(im)
        # lb = label_parser.convert_onehot_to_single_ch(lb)
        # blended = blend_images(im, lb, cmap="viridis", alpha=0.7)
        # blended = blended.numpy().transpose(1, 2, 0)
        # blended = (blended * 255).astype(np.uint8)

        axes[0].imshow(blended1)
        axes[0].axis("off")

        axes[1].imshow(blended2)
        axes[1].axis("off")

        break

    plt.show()


@hydra.main(version_base=None, config_path="../../config/phantom_seg", config_name="config")
def main(cfg: SegmentationConfig):
    print(OmegaConf.to_yaml(cfg))

    dataset_container = load_dataset(cfg)
    model = create_model(cfg, dataset_container)

    test_model(cfg, dataset_container, model)


if __name__ == "__main__":
    main()
