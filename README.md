# HerdNet
Code for paper "[From Crowd to Herd Counting: How to Precisely Detect and Count African Mammals using Aerial Imagery and Deep Learning?]()"

![](https://i.imgur.com/MCZWn8Z.jpg)



## Citation
If you use this code in your work, please cite our [paper]():
```
@article{}
```

## Pretrained models

 <font size="2">

| Model   | Params | Dataset                                                      | Species                                          | F1score | MAE¹ | RMSE² |  AC³  |                                           Download                                           |
| ------- |:------:| ------------------------------------------------------------ | ------------------------------------------------ |:-------:|:----:|:-----:|:-----:|:--------------------------------------------------------------------------------------------:|
| HerdNet |  18M   | Ennedi 2019                                                  | Camel, donkey, sheep and goat                    |  73.6%  | 6.1  |  9.8  | 15.8% | [PTH file](https://drive.google.com/uc?export=download&id=1CetqTS3VSilMI98Fx_u-yeNr7zBZYh94) |
| HerdNet |  18M   | [Delplanque et al. (2022)](https://doi.org/10.1002/rse2.234) | Buffalo, elephant, kob, topi, warthog, waterbuck |  83.5%  | 1.9  |  3.6  | 7.8%  | [PTH file](https://drive.google.com/uc?export=download&id=1-WUnBC4BJMVkNvRqalF_HzA1_pRkQTI_) |

¹MAE, Mean Absolute Error; ²RMSE, Root Mean Square Error; ³AC, Average Confusion between species.
    
</font>

Note that these metrics have been computed on full-size test images.

## Installation
Create and activate the conda environment
```console
conda env create -f environment.yml
conda activate animaloc
```

Install the code
```console
python setup.py install
```

Create a [Weights & Biases](https://wandb.ai/home) account and then log in
```console
wandb login
```

## Dataset Format
A CSV file which must contain the header **`images,x,y,labels`** for points, or **`images,x_min,y_min,x_max,y_max,y,labels`** for bounding boxes. Each row should represent one annotation, with at least, the image name (``images``), the object location within the image (`x`, `y`) for points, and (`x_min`, `y_min`, `x_max`, `y_max`) for bounding boxes and its label (`labels`):

Point dataset:
```csv
images,x,y,labels
Example.JPG,517,1653,2
Example.JPG,800,1253,1
Example.JPG,78,33,3
Example_2.JPG,896,742,1
...
```

Bounding boxe dataset:
```csv
images,x_min,y_min,x_max,y_max,labels
Example.JPG,530,1458,585,1750,4
Example.JPG,95,1321,152,1403,2
Example.JPG,895,478,992,658,1
Example_2.JPG,47,253,65,369,1
...
```

An image containing *n* objects is therefore spread over *n* lines.

## Quick Start
Set the seed for reproducibility
```python=
from animaloc.utils.seed import set_seed

set_seed(9292)
```

Create point datasets
```python=
import albumentations as A

from animaloc.datasets import CSVDataset
from animaloc.data.transforms import MultiTransformsWrapper, DownSample, PointsToMask, FIDT

patch_size = 512
num_classes = 4
down_ratio = 2

train_dataset = CSVDataset(
    csv_file = '/path/to/train/data.csv',
    root_dir = '/path/to/train/data',
    albu_transforms = [
        A.VerticalFlip(p=0.5), 
        A.Normalize(p=1.0)
        ],
    end_transforms = [MultiTransformsWrapper(
        FIDT(num_classes=num_classes, down_ratio=down_ratio),
        PointsToMask(radius=2, num_classes=num_classes, squeeze=True, down_ratio=int(patch_size//16))
        )]
    )

val_dataset = CSVDataset(
    csv_file = '/path/to/val/data.csv',
    root_dir = '/path/to/val/data',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )

```
Create dataloaders
```python=
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 4,
    shuffle = True
    )

val_dataloader = DataLoader(
    dataset = val_dataset,
    batch_size = 1,
    shuffle = False
    )
```
Instanciate HerdNet
```python=
from animaloc.models import HerdNet

herdnet = HerdNet(num_classes=num_classes, down_ratio=down_ratio)
```

Define the losses for training HerdNet
```python=
from torch import Tensor
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss

weight = Tensor([0.1, 1.0, 1.0, 1.0]).cuda()

losses = [
    {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
    {'loss': CrossEntropyLoss(reduction='mean', weight=weight), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
    ]

herdnet = LossWrapper(herdnet, losses=losses)
```

Train et validate HerdNet
```python=
from torch.optim import Adam
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator

work_dir = 'path/to/working/directory'

lr = 1e-4
weight_decay = 1e-3
epochs = 100

optimizer = Adam(params=herdnet.parameters(), lr=lr, weight_decay=weight_decay)

metrics = PointsMetrics(radius=5, num_classes=num_classes)

stitcher = HerdNetStitcher(
    model=herdnet, 
    size=(patch_size,patch_size), 
    overlap=160, 
    down_ratio=down_ratio, 
    reduction='mean'
    )

evaluator = HerdNetEvaluator(
    model=herdnet, 
    dataloader=val_dataloader, 
    metrics=metrics, 
    stitcher=stitcher, 
    work_dir=work_dir, 
    header='validation'
    )

trainer = Trainer(
    model=herdnet,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    num_epochs=epochs,
    evaluator=evaluator,
    work_dir=work_dir
    )

trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score')
```

Use pretrained model
```python=
from animaloc.models import HerdNet, LossWrapper, load_model

herdnet = HerdNet(num_classes=4, down_ratio=2)
herdnet = LossWrapper(herdnet, losses=[])
herdnet = load_model('path/to/the/file.pth')
```


## Tools
### Creating Patches
To train a model, such as HerdNet, it is often useful to extract patches from the original full-size images, especially if you have a GPU with limited memory. To do so, you can use the `patcher.py` tool:
```console
python tools/patcher.py <ROOT> <HEIGHT> <WIDTH> <OVERLAP> <DEST> -csv <CSV> -min <MIN> -all <BOOL>
```
For help, run:
```console
python tools/patcher.py -h
```

### Starting a Training Session
A training session can easily be launched using the `train.py` tool. This tool uses [Hydra](https://hydra.cc/) framework. You simply need to modify the basic config file and then run:
```console
python tools/train.py
```

You can also create your own config file. Save it first into the [`configs/train`](https://github.com/Alexandre-Delplanque/HerdNet/tree/main/configs/train) folder and then run:
```console
python tools/train.py train=<your config name>
```
Click [here](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/TRAINING_CONFIG.md) to see the description of the **training config file**.

You can also make multiple different configurations runs or modify some parameters directly from the command line (see the [doc](https://hydra.cc/docs/intro)).

### Starting a Testing Session
A testing session can easily be launched using the `test.py` tool. This tool uses [Hydra](https://hydra.cc/) framework again. You simply need to modify the basic config file and then run:
```console
python tools/test.py
```

You can also create your own config file. Save it first into the [`configs/test`](https://github.com/Alexandre-Delplanque/HerdNet/tree/main/configs/test) folder and then run:
```console
python tools/test.py test=<your config name>
```
Click [here](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/TESTING_CONFIG.md) to see the description of the **testing config file**.

### Visualizing Ground Truth (and Detections)
You can view your ground truth and your model's detections by using the `view.py` tool. This tool uses [FiftyOne](https://voxel51.com/fiftyone/). You simply need to specify a root directory that contains your images (`root`), your CSV file containing the ground truth (`gt`) and optionaly a CSV file containing model's detections (`-dets`). See dataset format below for your CSV files format.
```console
python tools/view.py root gt [-dets]
```