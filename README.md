# voting_traffic
This repository contains the code needed to run the simulations detailed in our paper, 
[Democratizing traffic control in smart cities](https://doi.org/10.1016/j.trc.2024.104511).


## Setup
To run this, please install this fork of CityFlow: https://github.com/mbkorecki/CityFlow

The following packages and versions are needed:
```
geopandas==0.12.1
gym==0.21
torch>=1.13
dill==0.3
```

Installation can be done using 
```
pip install -r requirements.txt
```

## Training
Training scripts may be found in `intersection_switching/run_train.py`. Use this to train the models needed.

## Running experiments
Training scripts may be found in `intersection_switching/run_vote.py`.


## Citing
If you have found any part of this work useful or relevant to your research, please cite our article:

```
@article{Korecki2024,
  title = {Democratizing Traffic Control in Smart Cities},
  doi = {10.1016/j.trc.2024.104511},
  journal = {Transportation Research Part C: Emerging Technologies},
  author = {Korecki,  Marcin and Dailisan,  Damian and Yang,  Joshua and Helbing,  Dirk},
  year = {2024},
  vol = {160},
  pages = {104511}
}
```
