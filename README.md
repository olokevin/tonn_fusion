# TOMFuN: A Tensorized Optical Multimodal Fusion Network

## Dependencies
* Python >= 3.6
* pyutils >= 0.0.1. See [pyutils](https://github.com/JeremieMelo/pyutility) for installation.
* pytorch-onn >= 0.0.1. See [pytorch-onn](https://github.com/JeremieMelo/pytorch-onn) for installation.
* Python libraries listed in `requirements.txt`
* NVIDIA GPUs and CUDA >= 10.2

## Data for Experiments

The processed data for the experiments (CMU-MOSI, IEMOCAP, POM) can be downloaded here:

https://drive.google.com/open?id=1CixSaw3dpHESNG0CaCJV6KutdlANP_cr

To run the code, you should download the pickled datasets and put them in the `data` directory.

Note that there might be NaN values in acoustic features, you could replace them with 0s.

## Usage

* TOMFuN Training 
  * `> cd ./fusion`
  * `> python train_iemocap.py configs/training/TOMFUN.yml`

* TOMFuN Inference
  `> cd ./PINNs/Black_Scholes`
  * Offline: `> python train_iemocap.py configs/inference/offline.yml --test_only`
  * Noise-aware Offline: `> python train_iemocap.py configs/inference/noise_aware_offline.yml --test_only`
  * Noise-aware On-chip: `> python train_iemocap.py configs/inference/noise_aware_on_chip.yml`