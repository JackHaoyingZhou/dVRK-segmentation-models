# Scripts documentation

The following page list all the tested and documented scripts. Other script not included in this list might not be working.

## Training scripts
Training scripts are located in the [scripts/train_scripts](../scripts/train_scripts/)

| Script                        | Description                                                |
|-------------------------------|------------------------------------------------------------|
| train_phantom_segmentation.py | Training and testing script using [Hydra][1] config files. |

## Utility scripts

| Script                        | Description                                     |
|-------------------------------|-------------------------------------------------|
| show_flexible_unet_summary.py | Print summary information of segmentation model |

## Data utilities

| Script                          | Description                                          |
|---------------------------------|------------------------------------------------------|
| Labelme_json_dataset_parsing.py | Parse labelme json files into rgb images and labels  |
| split_stereo_video.py           | Split combined left/right video into separete videos |
| image_from_video.py             | Sample images from video                             |

[1]: https://hydra.cc/
