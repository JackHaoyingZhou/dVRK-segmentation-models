# How to Run the Scripts for Training and Testing
Contributor: Juan Antonio Barragan & Haoying(Jack) Zhou

The overall structure of this repository should be as follows:
```
.
├── assets(not included)
│   ├── weights
│   │   ├── pretrained-weights
│   │   ├── ...
├── config
│   ├── phantom_instrument_seg(train)
│   ├── phantom_seg(test)
├── monai_data(not included)
│   ├── <your data>
│   ├── dataset_config.yaml
├── scripts
│   ├── train_scripts
│   │   ├── train_phantom_segmentation.py(train)
│   │   ├── test_phantom_segmentation_jack.py(test)
│   │   ├── ...
│   ├── copy_rename_files.py(data parser)
│   ├── create_binary_segment.py(create binary segment)
│   ├── ...
├── doc(README files)
├── surg_seg
├── ...
```

## Run Data Parser

```bash
python3 copy_rename_files.py -i <src_folder> -o <output_dir> -s <True or False>
```

This script will read all images in the `src_folder` then copy & rename them and save in the `output_dir`. `-s` Represent a status whether to split the dataset

** Important note, there is a configuration file called `dataset_config.yaml` included in the dataset root folder. If you don't have the file/ don't know how to create the file. Please feel free to submit an issue in the GitHub Repository.

| Command  | Meaning                 | Description                                                                                                           |
|----------|-------------------------|-----------------------------------------------------------------------------------------------------------------------|
| -i <...> | Source Folder of Images | Source Folder of Images, having structure shown in the following subsection                                           |
| -o <...> | Output Folder of Images | Output Folder of Images, having structure shown in the following subsection                                           |
| -s <...> | Flag of splitting data  | If true, it will split the data with 80% training and 20% validation subsets, otherwise, it will return 100% test set |


### Source Folder Structure

```
.
├── <...>
│   ├── rgb
│   │   ├── xxx.png
│   │   ├── ...
│   ├── segmented
│   │   ├── xxx.png
│   │   ├── ...
│   ├── ...
├── ...
```

### Output Folder Structure

#### -s True

```
.
├── train
│   ├── rgb
│   │   ├── xxx.png
│   │   ├── ...
│   ├── segmented
│   │   ├── xxx.png
│   │   ├── ...
├── valid
│   ├── rgb
│   │   ├── xxx.png
│   │   ├── ...
│   ├── segmented
│   │   ├── xxx.png
│   │   ├── ...
├── dataset_config.yaml
```

#### -s False

```
.
├── test
│   ├── rgb
│   │   ├── xxx.png
│   │   ├── ...
│   ├── segmented
│   │   ├── xxx.png
│   │   ├── ...
├── dataset_config.yaml
```

## Create Binary Segment

```bash
python3 create_binary_segment.py -i <src_folder>
```

This script converts the segmented images into binary segmented images. Then, it stores the binary segmentation at a subfolder named `binary_segmented` in the `src_folder`

## Train

```bash
cd train_scripts
python3 train_phantom_segmentation.py
```

Please check the xxx_jack.yaml for details

## Test

```bash
test_phantom_segmentation_jack.py
```

Please check the xxx_jack.yaml for details