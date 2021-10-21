## 1. The official repository for the project [ESRelation](https://hackerekcah.github.io/ESRelation):
> Paper Title: Exploring Inter-Node Relations in CNNs for Environmental Sound Classification
## 2. [&#9758; Project Page](https://hackerekcah.github.io/ESRelation) for dynamic visualization examples.
## 3. Trained Models
* click [here](https://pan.baidu.com/s/1s5gaF3mrcfp3_KeeYdfopw) (pw: `hack`) to download the models listed in the following tables.
### 3.1 ESC-50 dataset
| Model| 5Fold-CV |
| -----| -----|
ResNeXt-GMP (baseline) | 90.3|
ResNext-GMP + RBlock (our best) |91.9 | 

### 3.2 US8K dataset
| Model | 10fold-CV | 
| ----- | ----- |
| ResNeXt-GMP (baseline) | 84.7| 
| ResNeXt-GMP + RBlock (our best)| 85.9|

### 3.3 DCASE2018 Task1A dataset
| Model | Fold1 | 
| ----- | ----- |
| ResNeXt-GAP (baseline) | 77.24| 
| ResNeXt-GAP + RBlock (our best)| 79.15|
## 4. To reproduce the classification results in the paper
* Install python environment, see `requirements.yaml`
  * The code is based on `torch1.6.0` and `python3.6.10`
```bash
# needs to change the prefix in requirements.yaml file to your own directory
conda env create -f requirements.yaml
```
* Run with config files, under cfgs/ folders
``` python
python main.py --cfg <cfg-file>
```

## 5. To reproduce visualizations in the paper, see `vis_paper`

## 6. To reproduce visualizations in the project page, see `vis_proj`
## 7. Find this project interesting? [&#9758; See more papers on ESC.](https://hackerekcah.github.io/ESRelation/pub.html)

