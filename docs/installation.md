# Installation

The use all functionalities from this repository, you will need to install the package `surg_seg` and its dependencies.

## Dependencies

```bash
sudo apt install ffmpeg
```
## Installation of anaconda environment

```bash
conda create -n surg_env python=3.9 numpy ipython  -y && conda activate surg_env
```
## Installation of `surg_seg` package

`Surg_seg` is a python package that includes most of the code to interface with the trained models

```bash
pip install -e . -r requirements/requirements_basic.txt --user
```

The file `requirements_conda.txt` contains additional lightway rosdependencies. Do not install this dependencies on you base python interpreter, only on a virtual environment.


## Erase Anaconda virtual environment

```bash
conda remove --name surg_env --all
conda info --envs
```