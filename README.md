# kaggle-birdcall-resnet-baseline-training

The repository to show the way I trained https://www.kaggle.com/hidehisaarai1213/birdcall-resnet50-init-weights

## Prerequisites

* Python >= 3.6
* ffmpeg
* sox
* GPU (P100 / V100) is the best

## Intstallation

`pip install -r requirements.txt`

## To reproduce

1. First you need to put dataset in `/input/birdsong-recognition`. The data structure should be the same as that of Kaggle dataset.
2. Run `make prepare` at the top of this repository. This command will perform resampling the dataset to 32kHz. Note that this will take a bit long (1 Hours on NVIDIA DGX Station).
3. Run `make train` at the top of this repository.
