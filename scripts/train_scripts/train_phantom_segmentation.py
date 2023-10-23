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

##################################################################
# Concrete implementation of abstract classes
##################################################################
class BopImageDirParser(ImageDirParser):
    def __init__(self, root_dirs: List[Path]):
        super().__init__(root_dirs)

        self.parse_image_dir(root_dirs[0])

    def parse_image_dir(self, root_dir: Path):
        self.images_list = natsorted(list(root_dir.glob("*/rgb/*.png")))
        self.labels_list = natsorted(list(root_dir.glob("*/segmented/*.png")))


@dataclass
class DatasetContainer:
    label_parser: SegmentationLabelParser
    ds_train: ImageSegmentationDataset
    dl_train: ThreadDataLoader
    ds_test: ImageSegmentationDataset
    dl_test: ThreadDataLoader


##################################################################
# Auxiliary functions
##################################################################
def create_label_parser(config: SegmentationConfig) -> SegmentationLabelParser:
    mapping_file = Path(config.path_config.mapping_file_path)
    assert mapping_file.exists(), f"Mapping file {mapping_file} does not exist"

    label_info_reader = YamlSegMapReader(mapping_file)
    label_parser = SegmentationLabelParser(label_info_reader)

    return label_parser


def create_dataset_and_dataloader(
    config: ConfigParser, label_parser: SegmentationLabelParser, batch_size: int, split: str
) -> Tuple[ImageSegmentationDataset, ThreadDataLoader]:
    assert split in ["train", "test"], "Split must be either train or test"

    if split == "train":
        data_dir = Path(config.path_config.train_data_path)
    elif split == "test":
        data_dir = Path(config.path_config.test_data_path)

    data_reader = BopImageDirParser([data_dir])

    if split == "train":
        ds = ImageSegmentationDataset(
            label_parser,
            data_reader,
            color_transforms=ImageTransforms.img_transforms,
            geometric_transforms=ImageTransforms.geometric_transforms,
        )
    elif split == "test":
        ds = ImageSegmentationDataset(
            label_parser, data_reader, color_transforms=ImageTransforms.img_transforms
        )
    dl = ThreadDataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=True)

    return ds, dl


def train_with_image_dataset(
    config: SegmentationConfig, data_container: DatasetContainer, model: FlexibleUNet
):
    device = config.train_config.device

    # Load trainer
    training_output_path = Path(config.path_config.trained_weights_file_path)
    epochs = config.train_config.epochs
    learning_rate = config.train_config.learning_rate

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    trainer = ModelTrainer(device=device, max_epochs=epochs)
    model, training_stats = trainer.train_model(
        model, optimizer, data_container.dl_train, validation_dl=data_container.dl_test
    )

    # Save model
    training_output_path.mkdir(exist_ok=True)
    torch.save(
        model.state_dict(), training_output_path / config.path_config.trained_weights_file_name
    )
    training_stats.to_pickle(training_output_path)
    training_stats.plot_stats(file_path=training_output_path)

    print(f"Last train IOU {training_stats.iou_list[-1]}")
    print(f"Last validation IOU {training_stats.validation_iou_list[-1]}")

    return model


def show_images(dl: ThreadDataLoader, label_parser: SegmentationLabelParser) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.set_tight_layout(True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i, ax in enumerate(axes.flat):
        pair = next(iter(dl))
        im = pair["image"][0]
        lb = pair["label"][0]

        im = ImageTransforms.inv_transforms(im)
        lb = label_parser.convert_onehot_to_single_ch(lb)
        blended = blend_images(im, lb, cmap="viridis", alpha=0.7)
        blended = blended.numpy().transpose(1, 2, 0)
        blended = (blended * 255).astype(np.uint8)
        ax.imshow(blended)
        ax.axis("off")

    plt.show()


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
        print(blended.shape)
        Image.fromarray(blended).save(predictions_dir / pred_name)

        # im = ImageTransforms.inv_transforms(im)
        # lb = label_parser.convert_onehot_to_single_ch(lb)
        # blended = blend_images(im, lb, cmap="viridis", alpha=0.7)
        # blended = blended.numpy().transpose(1, 2, 0)
        # blended = (blended * 255).astype(np.uint8)

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

        fig, ax = plt.subplots(1, 1)
        ax.imshow(blended)
        # ax.imshow(np.transpose(img, (1, 2, 0)))
        x, y = np.where(single_ch_prediction[0].numpy() == 1)
        if len(x) == 0:
            print(f"{x[0]}, {y[0]}, len: {len(x)}")
        else:
            print("no needle")
        plt.show()
        print("hello")

    iou_stats.calculate_aggregated_stats()
    table = AggregatedMetricTable(iou_stats)
    table.fill_table()
    table.print_table()


def load_dataset(cfg: SegmentationConfig) -> DatasetContainer:
    label_parser = create_label_parser(cfg)
    ds_train, dl_train = create_dataset_and_dataloader(
        cfg, label_parser, batch_size=cfg.train_config.batch_size, split="train"
    )
    ds_test, dl_test = create_dataset_and_dataloader(
        cfg, label_parser, batch_size=cfg.test_config.batch_size, split="test"
    )

    print(f"Training dataset size: {len(ds_train)}")
    print(f"Validation dataset size: {len(ds_test)}")
    print(f"Number of output clases: {label_parser.mask_num}")

    dataset_container = DatasetContainer(label_parser, ds_train, dl_train, ds_test, dl_test)
    return dataset_container


def create_model(cfg: SegmentationConfig, dataset_container: DatasetContainer) -> FlexibleUNet:
    pretrained_weights_path = cfg.train_config.pretrained_weights_path
    model = create_FlexibleUnet(
        cfg.train_config.device, pretrained_weights_path, dataset_container.label_parser.mask_num
    )
    return model


def test_model(
    config: SegmentationConfig, dataset_container: DatasetContainer, model: FlexibleUNet
):
    path2weights = Path(config.path_config.trained_weights_file_path)
    path2weights /= config.path_config.trained_weights_file_name

    label_parser = dataset_container.label_parser
    ds = dataset_container.ds_test

    model_pipe = FlexibleUnet1InferencePipe(
        path2weights, config.test_config.device, out_channels=label_parser.mask_num, model=model
    )
    model_pipe.model.eval()
    model_pipe.upload_weights()

    if config.actions.show_inferences:
        save_test_predictions(config, dataset_container, model_pipe)

    if config.actions.calculate_metrics:
        calculate_metrics_on_valid(config, dataset_container, model_pipe)


##################################################################
# Main functions
##################################################################

cs = ConfigStore.instance()
cs.store(name="base_config", node=SegmentationConfig)


@hydra.main(
    version_base=None,
    config_path="../../config/segmentation_models",
    config_name="phantom_seg",
)
def main(cfg: SegmentationConfig):
    print(OmegaConf.to_yaml(cfg))

    dataset_container = load_dataset(cfg)
    model = create_model(cfg, dataset_container)

    if cfg.actions.show_images:
        show_images(dataset_container.dl_train, dataset_container.label_parser)

    if cfg.actions.train:
        model = train_with_image_dataset(cfg, dataset_container, model)

    if cfg.actions.test:
        test_model(cfg, dataset_container, model)

    from hydra.core.hydra_config import HydraConfig

    print(HydraConfig.get().job.config_name)


if __name__ == "__main__":
    main()
