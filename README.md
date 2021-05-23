## 1. The official repository for the project [ESRelation](https://hackerekcah.github.io/ESRelation):
> Paper Title: Exploring Inter-Node Relations in CNNs for Environmental Sound Classification
## 2. [&#9758; Project Page](https://hackerekcah.github.io/ESRelation) for dynamic visualization examples.
## 3. Trained Models
* click [here](https://pan.baidu.com/s/1s5gaF3mrcfp3_KeeYdfopw) (pw: `hack`) to download the models listed in the following tables.
### 3.1 ESC-50 dataset
| Model| Fold1| Fold2 | Fold3 | Fold4 | Fold5 |
| -----| -----| ----- | ----- | ----- | ----- |
ResNeXt-GMP (baseline) | 88.50| 89.50| 91.00| 92.75| 89.75|
ResNext-GMP + RBlock (our best) |90.75 | 91.25| 91.75| 94.00| 91.50|

### 3.2 DCASE2018 Task1A dataset
| Model | Fold1 | 
| ----- | ----- |
| ResNeXt-GAP (baseline) | 77.24| 
| ResNeXt-GAP + RBlock (our best)| 78.87|
## 4. To reproduce the classification results in the paper
* Install python environment, see `requirements.yaml`
  * The code is based on `torch1.6.0` and `python3.6.10`
```bash
# needs to change the prefix in requirements.yaml file to your own directory
# this will create a conda environment named "torch151"
conda env create -f requirements.yaml
```
* Run with config files
``` python
python main.py --cfg <cfg-file>

# ResNeXt-GMP (baseline) model on ESC-50
python main.py --cfg cfgs/esc/esc_folds_baseline.yaml

# ResNeXt-GMP + RBlock model on ESC-50
# different R-Blcok can be constructed by using different r_structure_type and softmax_type
python main.py --cfg cfgs/esc/esc_folds_rblock_pe.yaml

# ResNeXt-GAP (baseline) model on DCASE2018 Task1A dataset
python main.py --cfg cfgs/dcase/dcase_folds_baseline.yaml

# ResNeXt-GAP + RBlock model on DCASE2018 Task1A dataset
# different R-Blcok can be constructed by using different r_structure_type and softmax_type
python main.py --cfg cfgs/dcase/dcase_folds_rblock_pe.yaml
```

## 5. To reproduce visualizations in the paper, see `vis_paper`

## 6. To reproduce visualizations in the project page, see `vis_proj`
## 7. Find this project interesting? [&#9758; See more papers on ESC.](https://hackerekcah.github.io/ESRelation/pub.html)

