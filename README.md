# EnsembleMOT


> ***EnsembleMOT: A Step towards Ensemble Learning of Multiple Object Tracking***
>
> Yunhao Du, Zihang Liu, Fei Su
>
> [*arxiv 2210.05278*](https://arxiv.org/abs/2210.05278)

## How to run

Just run `python EnsembleMOT.py` directly.

You can modify the `methods` list to select trackers to be merged.

For example, the default is 
```python
methods = [
    join(dir_results, 'FairMOT'),
    join(dir_results, 'SiamMOT'),
]
```

and you can change it to 
```python
methods = [
    join(dir_results, 'TransTrack'),
    join(dir_results, 'CenterTrack'),
]
```

## Visualization
![image](https://github.com/dyhBUPT/EnsembleMOT/assets/99722489/b3998581-7b53-4afa-bba9-3286accc0847)


## Results
![image](https://github.com/dyhBUPT/EnsembleMOT/assets/99722489/c3e2a8cb-4a6a-4fda-b1e2-fed060471d03)


## TODO
- Voting-based Ensemble
- DeepSORT-based Ensemble

## Citation
```
@misc{2210.05278,
Author = {Yunhao Du and Zihang Liu and Fei Su},
Title = {EnsembleMOT: A Step towards Ensemble Learning of Multiple Object Tracking},
Year = {2022},
Eprint = {arXiv:2210.05278},
}
```
