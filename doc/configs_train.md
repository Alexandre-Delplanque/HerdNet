# Training config file

Here are the basics for creating a training config file as well as explanations of the parameters to be specified. The file must be in **YAML format** and be placed in the [`configs/train`](https://github.com/Alexandre-Delplanque/HerdNet/tree/main/configs/train) folder.

The config file is composed of **5 main parts**, which are detailed below:
```yaml
# 1) experiments settings
wandb_project: ...
wandb_entity: ...
seed: ...
device_name: ...
# 2) model
model: ...
# 3) losses
losses: ...
# 4) datasets
datasets: ...
# 5) training settings
training_settings: ...
```


---


## Experiment settings
First, you need to set your experiment by specifying your Weights & Biaises project, the seed and the device on which you want to train your model:
```yaml
wandb_project: 'my_project'
wandb_entity: 'my_name'
wandb_run: 'my_run'
seed: 100
device_name: 'cuda'
```
If the project does not exist yet, it will be automatically created in your account, in the specified entity.

## Model and Losses
Then, you need to define your model and the loss(es) that will be used for training. You also have to mention the name of your `nn.Module` class (previously defined in [`animaloc.models`](https://github.com/Alexandre-Delplanque/HerdNet/tree/main/animaloc/models)).

You can either choose to start your training session with pretrained parameters stored into a PTH file (`load_from`) or to resume training from a checkpoint (`resume_from`).

```yaml
model: 
  name: 'MyModel'      # nn.Module class name
  load_from: null      # path to your trained parameters (.pth)
  resume_from: null    # path to your trained parameters (.pth)
  kwargs:              # model's keyword arguments
    num_layers: 34
    img_size: [512,512]
    pretrained: True
    down_ratio: 2
    head_conv: 256
```

Internally, the model is then wrapped by a [LossWrapper](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/animaloc/models/utils.py#L63) class to add loss(es) for training. This is specified as follows: 

```yaml
losses:
  FocalLoss:
    print_name: 'focal_loss'
    from_torch: False
    output_idx: 0
    target_idx: 0
    lambda_const: 1.0
    kwargs: # loss' keyword arguments
      reduction: 'mean'
```

Each loss must be specified by its class name as key, and must contain 5 mandatory parameters:

* **`print_name`**, the name (string) that will be used to print the loss value during training,
* **`from_torch`**, a flag (boolean) used to specify whether the loss comes from PyTorch (True) or if it is a personal one (False). In this case, the loss must be saved in the [losses folder](https://github.com/Alexandre-Delplanque/HerdNet/tree/main/animaloc/train/losses),
* **`output_idx`**, the output index (int) of the model, which will be used for the loss computation (useful only when there are several outputs, as in the case of Animaloc for example),
* **`target_idx`**, the target index (int), which will be used for the loss computation (useful only when there are several targets, as in the case of Animaloc for example),
* **`lambda_const`**, a loss weighting value (float).

## Datasets parameters
The following section defines the datasets that will be used during training and model validation.

### Overview

First, you need to give some basics parameters:
```yaml
datasets:
  image_size: [512,512]    # the images size
  anno_type: 'point'       # the annotations type
  num_classes: 4           # the number of classes, background included
  collate_fn: null         # the collate function used by dataloaders, should be 
                           # defined in animaloc.data.batch_utils.py
...
```
Then, define your labels and class names. This will be used to ensure the correct matching between predicted labels and true class names.
```yaml 
...
  class_def:
    1: 'camel'
    2: 'donkey'
    3: 'sheep/goat'
...
```

Finally, define the datasets. The key **`name`** is used to specify the dataset you want to use. The available datasets can be found [here](https://github.com/Alexandre-Delplanque/HerdNet/tree/main/animaloc/datasets). The datasets are defined mainly by a CSV file containing the annotations (`csv_file`), a root directory containing the images (`root_dir`), [Albumentations](https://albumentations.ai/) transformations (`albu_transforms`) and end transformations (`end_transforms`).

```yaml
...
  train:
    name: 'CSVDataset'
    csv_file: '/your/path/train.csv'
    root_dir: '/your/path/train'
    
    sample_on: null

    albu_transforms:
      HorizontalFlip:
        p: 0.5
      Normalize:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    
    end_transforms:
      PointsToMask:
        radius: 5
        num_classes: 2
        down_ratio: 2

  validate:
    name: 'CSVDataset'
    csv_file: '/your/path/validation.csv'
    root_dir: '/your/path/val'

    albu_transforms:
      Normalize:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    
    end_transforms:
      PointsToMask:
        radius: 5
        num_classes: 2
        down_ratio: 2
```

### Albumentations transforms

The **`albu_transforms`** key is used to list all the [Albumentations](https://albumentations.ai/) transformations you want to use on your dataset. A list of available transformations can be found [here](https://albumentations.ai/docs/getting_started/transforms_and_targets/). To specify one of them, you just have to specify its name as a key, and fill in its keyword arguments:
```yaml
albu_transforms:
  HorizontalFlip:                 # name of your first transformation
    p: 0.5                        # kwargs
  Normalize:                      # name of your second transformation
    mean: [0.485, 0.456, 0.406]   # kwargs
    std: [0.229, 0.224, 0.225]
```

### End transforms

The **`end_transforms`** key is used to list all additional transformations to be applied. They must be custom transforms, stored in the [`transforms.py`](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/animaloc/data/transforms.py) module, and can be used to switch from 1D data (list of coordinates) to 2D (map), for instance. The available transformations can be found [here](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/animaloc/data/transforms.py). As for the Albumentations transformations, you just have to specify its name as a key, and fill in its keyword arguments:
```yaml
end_transforms:
  PointsToMask:        # name of your first transformation
    radius: 5          # kwargs
    num_classes: 2
    down_ratio: 2
```
In case no end transforms are required:
```yaml
end_transforms: null
```

## Training settings

The last section of the config file relates to the training settings.

### Overview
Most of the keys are named explicitly for their purpose:
```yaml
training_settings:
  print_freq: 50           # print frequency
  valid_freq: 1            # validation frequency (epochs)
  batch_size: 8            # batch size
  optimizer: 'adam'        # optimizer: choose between 'adam' or 'sgd'
  lr: 1e-4                 # learning rate
  weight_decay: 0.0005     # weight decay
  auto_lr: False           # option for automatic LR scheduler
  warmup_iters: 50         # number of warmup iterations
  vizual_fn: null          # name of a plotting function used during validation,
                           # should be defined in animaloc.vizual.plots
  epochs: 100              # number of epochs
  evaluator: null          # option for the validation dataset evaluation
  stitcher: null           # option for using a stitcher during validation
```

### Automatic LR scheduler

The **`auto_lr`** key allows to use the [`ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau) scheduler of PyTorch. This automates the reduction of the learning rate during the training according to the evolution of the losses values of the validation set or following the evolution of a metric computed on it (see evaluator below). For more information, see the doc [here](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau).

To use this option, simply enter the keywords arguments and their respective values:
```yaml
auto_lr:                 
  mode: 'max'
  patience: 10
  threshold: 1e-4
  threshold_mode: 'abs'
  min_lr: 1e-7
  verbose: True
```

### Evaluator

You can specify a custom evaluator that will be used for model validation:

```yaml
evaluator: 
  name: 'Evaluator'
  threshold: 10
  select_mode: 'max'
  validate_on: 'mAP'
  kwargs:
    print_freq: 50
```

To do this, you need to specify 4 parameters:

* **`name`**, class name for the evaluator to be used.
* **`threshold`**, threshold (float) used to define a True Positive (e.g. a radius value in case of point model evaluation),
* **`select_mode`**, best model selection mode (string): 
    * `'min'`, for selecting the epoch that yields to a minimum validation value,
    * `'max'`, for selecting the epoch that yields to a maximum validation value.
* **`validate_on`**, metrics used for validation (string). Possible values are: `'recall'`, `'precision'`, `'f1_score'`, `'mse'`, `'mae'`, `'rmse'` and `'mAP'`.
* **`kwargs`**, evaluator's keyword arguments.

### Stitcher
When specified, used to perform inference on large images. The module works as a moving window on the large image. It collects the predictions of each patch and aggregates them to get the detections in the coordinate system of the whole image. For more information, check the [`Stitcher`](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/animaloc/eval/stitchers.py#L38) class.
```yaml
stitcher:
  name: 'Stitcher'        # class name
  kwargs:                 # keyword arguments
    overlap: 100
    down_ratio: 2
    up: False
    reduction: 'mean'
```


---

## Tips

### Same value for multiple parameters
In case a parameter value is used in several parts of the config file, it is possible to refer to it by typing:
```yaml
param: ${train.<part>.<parameter>}
```
For instance:
```yaml
end_transforms:
  PointsToMask:        # name of your first transformation
    radius: 5          # kwargs
    num_classes: 2
    down_ratio: ${train.model.kwargs.down_ratio}
```

### Multiple runs
If you want to run the same training session with different configs, e.g. by varying the values of a hyperparameter, you can use the multi-run flag of Hydra: `-m`. To do so, pass a comma separated list specifying the values for each dimension you want to sweep, e.g.:
```console
python tools/train.py -m train=animaloc train.training_settings.batch_size=2,4,8 train.training_settings.lr=1e-3,1e-4
```
This will sweep over all combinations of specified batch sizes and learning rates (here 6 combinations, so 6 training sessions). 