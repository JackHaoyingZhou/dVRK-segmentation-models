from pathlib import Path
import random
from natsort import natsorted
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import monai
from tqdm import tqdm, trange
from monai.visualize.utils import blend_images
import torch
import torchvision.transforms as T
from monai.bundle import ConfigParser
from monai.data import ThreadDataLoader
from monai.networks.nets import FlexibleUNet
import monai.transforms as mt
from torchvision import transforms
from torchvision.transforms import functional as TF

from surg_seg.Datasets.SegmentationLabelParser import (
    SegmentationLabelParser,
    YamlSegMapReader,
)
from surg_seg.Datasets.ImageDataset import (
    ImageDirParser,
    ImageSegmentationDataset,
)
from surg_seg.Metrics.MetricsUtils import AggregatedMetricTable, IOUStats
from surg_seg.Networks.Models import FlexibleUnet1InferencePipe
from surg_seg.Trainers.Trainer import ModelTrainer


def create_FlexibleUnet(device, pretrained_weights_path: Path, out_channels: int):
    model = FlexibleUNet(
        in_channels=3,
        out_channels=out_channels,
        backbone="efficientnet-b0",
        pretrained=True,
        is_pad=False,
    ).to(device)

    pretrained_weights = monai.bundle.load(
        name="endoscopic_tool_segmentation", bundle_dir=pretrained_weights_path, version="0.2.0"
    )
    model_weight = model.state_dict()
    weights_no_head = {k: v for k, v in pretrained_weights.items() if not "segmentation_head" in k}
    model_weight.update(weights_no_head)
    model.load_state_dict(model_weight)

    return model


class ImageTransforms:

    img_transforms_train = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalize
            # T.RandomCrop((480, 640)),
            # T.RandomVerticalFlip(),
            # T.RandomHorizontalFlip(),
            # T.RandomInvert(0.5),
            # T.ColorJitter(),
            # T.RandomGrayscale(0.1),
        ]
    )
    img_transforms_valid = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalize
        ]
    )
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    inv_transforms = T.Compose(
        [
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    def geometric_transforms(image, mask):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(480, 640))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    predictions_trans = mt.Compose(
        [
            mt.Activations(sigmoid=True),
            mt.AsDiscrete(threshold=0.5),
        ]
    )


class CustomImageDirParser(ImageDirParser):
    def __init__(self, root_dirs: List[Path]):
        super().__init__(root_dirs)

        self.parse_image_dir(root_dirs[0])

    def parse_image_dir(self, root_dir: Path):
        self.images_list = natsorted(list((root_dir / "raw").glob("*.png")))
        self.labels_list = natsorted(list((root_dir / "label").glob("*.png")))


def train_with_image_dataset(config: ConfigParser):
    train_config = config.get_parsed_content("ambf_train_config")

    train_dir_list = train_config["train_dir_list"]
    valid_dir_list = train_config["val_dir_list"]
    pretrained_weights_path = train_config["pretrained_weights_path"]
    training_output_path = train_config["training_output_path"]
    mapping_file = train_config["mapping_file"]

    device = train_config["device"]
    epochs = train_config["epochs"]
    learning_rate = train_config["learning_rate"]

    # Train model
    train_data_reader = CustomImageDirParser(train_dir_list)
    valid_data_reader = CustomImageDirParser(valid_dir_list)
    label_info_reader = YamlSegMapReader(mapping_file)
    label_parser = SegmentationLabelParser(label_info_reader)

    ds = ImageSegmentationDataset(
        label_parser,
        train_data_reader,
        color_transforms=ImageTransforms.img_transforms_train,
        geometric_transforms=ImageTransforms.geometric_transforms,
    )
    dl = ThreadDataLoader(ds, batch_size=8, num_workers=2, shuffle=True)

    val_ds = ImageSegmentationDataset(
        label_parser, valid_data_reader, color_transforms=ImageTransforms.img_transforms_valid
    )
    val_dl = ThreadDataLoader(val_ds, batch_size=8, num_workers=2, shuffle=True)

    print(f"Training dataset size: {len(ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Number of output clases: {label_parser.mask_num}")

    model = create_FlexibleUnet(device, pretrained_weights_path, label_parser.mask_num)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    trainer = ModelTrainer(device=device, max_epochs=epochs)
    model, training_stats = trainer.train_model(model, optimizer, dl, validation_dl=val_dl)

    training_output_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), training_output_path / "myweights.pt")
    training_stats.to_pickle(training_output_path)
    training_stats.plot_stats(file_path=training_output_path)

    print(f"Last train IOU {training_stats.iou_list[-1]}")
    print(f"Last validation IOU {training_stats.validation_iou_list[-1]}")


def show_images(config: ConfigParser, show_valid: str = False):
    train_config = config.get_parsed_content("ambf_train_config")
    mapping_file = train_config["mapping_file"]
    if show_valid:
        dir_list = train_config["val_dir_list"]
    else:
        dir_list = train_config["train_dir_list"]

    # Train model
    train_data_reader = CustomImageDirParser(dir_list)
    label_info_reader = YamlSegMapReader(mapping_file)
    label_parser = SegmentationLabelParser(label_info_reader)

    ds = ImageSegmentationDataset(
        label_parser,
        train_data_reader,
        color_transforms=ImageTransforms.img_transforms_train,
        geometric_transforms=ImageTransforms.geometric_transforms,
    )
    dl = ThreadDataLoader(ds, batch_size=1, num_workers=0, shuffle=True)

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


def inference_on_valid(config: ConfigParser):
    device = "cuda"
    mapping_file = config.get_parsed_content("ambf_train_config#mapping_file")
    valid_dir_list = config.get_parsed_content("ambf_train_config#val_dir_list")
    path2weights = config.get_parsed_content("test#weights")

    train_data_reader = CustomImageDirParser(valid_dir_list)
    label_info_reader = YamlSegMapReader(mapping_file)
    label_parser = SegmentationLabelParser(label_info_reader)

    ds = ImageSegmentationDataset(
        label_parser,
        train_data_reader,
        color_transforms=ImageTransforms.img_transforms_train,
    )
    dl = ThreadDataLoader(ds, batch_size=1, num_workers=0, shuffle=True)

    model_pipe = FlexibleUnet1InferencePipe(
        path2weights, device, out_channels=label_parser.mask_num
    )
    model_pipe.model.eval()

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.set_tight_layout(True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i, ax in enumerate(axes.flat):
        # pair = next(iter(dl))
        pair = ds.__getitem__(i, transform=False)
        im = pair["image"]
        lb = pair["label"]
        print(im.shape)
        input_tensor, inferred_single_ch = model_pipe.infer(im)

        inferred_single_ch = inferred_single_ch.detach().cpu()
        input_tensor = input_tensor.detach().cpu()[0]
        blended = blend_images(input_tensor, inferred_single_ch, cmap="viridis", alpha=0.8).numpy()
        blended = (np.transpose(blended, (1, 2, 0)) * 254).astype(np.uint8)

        # im = ImageTransforms.inv_transforms(im)
        # lb = label_parser.convert_onehot_to_single_ch(lb)
        # blended = blend_images(im, lb, cmap="viridis", alpha=0.7)
        # blended = blended.numpy().transpose(1, 2, 0)
        # blended = (blended * 255).astype(np.uint8)
        ax.imshow(blended)
        ax.axis("off")
    plt.show()


def calculate_metrics_on_valid(config: ConfigParser):
    device = "cuda"
    mapping_file = config.get_parsed_content("ambf_train_config#mapping_file")
    valid_dir_list = config.get_parsed_content("ambf_train_config#val_dir_list")
    path2weights = config.get_parsed_content("test#weights")

    train_data_reader = CustomImageDirParser(valid_dir_list)
    label_info_reader = YamlSegMapReader(mapping_file)
    label_parser = SegmentationLabelParser(label_info_reader)

    ds = ImageSegmentationDataset(
        label_parser,
        train_data_reader,
        color_transforms=ImageTransforms.img_transforms_train,
    )
    dl = ThreadDataLoader(ds, batch_size=1, num_workers=2, shuffle=True)

    model_pipe = FlexibleUnet1InferencePipe(
        path2weights, device, out_channels=label_parser.mask_num
    )
    model_pipe.model.eval()

    iou_stats = IOUStats(label_parser)
    for batch in tqdm(dl, desc="Calculating metrics"):
        img = batch["image"].to(device)
        label = batch["label"]
        img_paths = ["empty"] * label.shape[0]

        prediction = model_pipe.model(img).detach().cpu()
        onehot_prediction = ImageTransforms.predictions_trans(prediction)
        iou_stats.calculate_metrics_from_batch(onehot_prediction, label, img_paths)

    iou_stats.calculate_aggregated_stats()
    table = AggregatedMetricTable(iou_stats)
    table.fill_table()
    table.print_table()


def main():
    # Config parameters
    config = ConfigParser()
    config.read_config("./training_configs/juanubuntu/dvrk_train_config.yaml")

    # show_images(config, show_valid=True)
    # train_with_image_dataset(config)
    # inference_on_valid(config)
    calculate_metrics_on_valid(config)


if __name__ == "__main__":
    main()
