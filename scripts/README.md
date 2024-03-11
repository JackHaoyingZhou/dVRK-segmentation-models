# How to Run the Scripts for Training and Testing
Contributor: Juan Antoni Barragan & Haoying(Jack) Zhou

## Run Data Parser

```bash
python3 copy_rename_files.py -i <src_folder> -o <output_dir> -s <True or False>
```

This script will read all images in the `src_folder` then copy & rename them and save in the `output_dir`. `-s` Represent a status whether to split the dataset

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
python3 create_binary_segment.py -i <src_foler>
```

## Train

```bash
cd train_scripts
python3 train_phantom_segmentation.py
```

## Test

```bash
test_phantom_segmentation_jack.py
```