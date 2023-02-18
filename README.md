# dynamicslearner

Deep Learning based extension of "Contact Observer for Humanoid Robot Pepper based on Tracking Joint Position Discrepancies", Bolotnikova et al. RO-MAN 2018.
TL;DR: Estimating the motor dynamics of the Pepper robot is difficult as a lot of the constants/coefficients are unknown or difficult to estimate. Therefore, Machine Learning can be used to learn the desired Joint Position Discrepancies between the actual joint positions and control outputs (desired position, velocity, acceleration and torque).

## Dependencies

- [`mc_rtc` (v1.6.0)](https://github.com/jrl-umi3218/mc_rtc/releases/tag/v1.6.0)
- [`pytorch`](https://pytorch.org)
- [`scikit-learn`](https://scikit-learn.org)
- [`matplotlib`](https://matplotlib.org/)

## Preprocessing

The raw data collected from random trajectories executed on Pepper using the `mc_rtc` framework is stored in [`data/raw/bins`](`data/raw/bins`), from which csv files are extracted by running [`bin_extractor.sh`](bin_extractor.sh) into the folder [`data/raw/csv`](`data/raw/csv`).
From this, the data for the Right arm joints (RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll) are extracted into numpy format by running [`preproc.py`](preproc.py) into [`data/proc/preprocessed.npz`](data/proc/preprocessed.npz) which has 2 keys: `"inputs"` and `"targets"` which are used to train the network accordingly.
