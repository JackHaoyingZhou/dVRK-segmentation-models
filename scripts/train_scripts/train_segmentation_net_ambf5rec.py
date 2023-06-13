from pathlib import Path
from tqdm import trange
import torch

import monai
from monai.bundle import ConfigParser
from monai.data import ThreadDataLoader
from monai.networks.nets import FlexibleUNet

from surg_seg.Datasets.SegmentationLabelParser import Ambf5RecSegMapReader, SegmentationLabelParser
from surg_seg.Datasets.ImageDataset import Ambf5RecDataReader, ImageSegmentationDataset
from surg_seg.Datasets.VideoDatasets import CombinedVidDataset
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


def train_with_video_dataset():
    device = "cpu"

    vid_root = Path("/home/juan1995/research_juan/accelnet_grant/data/rec01/")
    vid_filepath = vid_root / "raw/rec01_seg_raw.avi"
    seg_filepath = vid_root / "annotation2colors/rec01_seg_annotation2colors.avi"
    ds = CombinedVidDataset(vid_filepath, seg_filepath)
    dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

    pretrained_weigths_path = Path("./assets/weights/trained-weights.pt")
    model = create_FlexibleUnet(device, pretrained_weigths_path, ds.label_channels)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)

    trainer = ModelTrainer(device=device, max_epochs=2)
    model, training_stats = trainer.train_model(model, optimizer, dl)

    training_stats.plot_stats()

    model_path = "./assets/weights/myweights_video"
    torch.save(model.state_dict(), model_path)
    training_stats.to_pickle(model_path)


def train_with_image_dataset():
    # Config parameters
    config = ConfigParser()
    config.read_config("./training_configs/juanubuntu/ambf_train_config.yaml")
    train_config = config.get_parsed_content("ambf_train_config")

    train_dir_list = train_config["train_dir_list"]
    valid_dir_list = train_config["val_dir_list"]
    annotations_type = train_config["annotations_type"]
    pretrained_weights_path = train_config["pretrained_weights_path"]
    training_output_path = train_config["training_output_path"]
    mapping_file = train_config["mapping_file"]

    device = train_config["device"]
    epochs = train_config["epochs"]
    learning_rate = train_config["learning_rate"]

    # Train model
    train_data_reader = Ambf5RecDataReader(train_dir_list, annotations_type)
    valid_data_reader = Ambf5RecDataReader(valid_dir_list, annotations_type)
    label_info_reader = Ambf5RecSegMapReader(mapping_file, annotations_type)
    label_parser = SegmentationLabelParser(label_info_reader)

    ds = ImageSegmentationDataset(label_parser, train_data_reader)
    dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

    val_ds = ImageSegmentationDataset(label_parser, valid_data_reader)
    val_dl = ThreadDataLoader(val_ds, batch_size=4, num_workers=0, shuffle=True)

    print(f"Training dataset size: {len(ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

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


def main():
    # train_with_video_dataset()
    train_with_image_dataset()


if __name__ == "__main__":
    main()
