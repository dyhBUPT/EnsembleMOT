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

![image](https://user-images.githubusercontent.com/99722489/196875661-ab7d2a94-b776-459d-9f88-bd7920b6545b.png)

## Results

![image](https://user-images.githubusercontent.com/99722489/196875733-32163421-c7f5-42da-b972-f69e09c5891a.png)

![image](https://user-images.githubusercontent.com/99722489/196875839-54f60b58-252b-447e-a9f0-20c461077335.png)


## Citation
```
@misc{2210.05278,
Author = {Yunhao Du and Zihang Liu and Fei Su},
Title = {EnsembleMOT: A Step towards Ensemble Learning of Multiple Object Tracking},
Year = {2022},
Eprint = {arXiv:2210.05278},
}
```
