# Testing config file

Here are the basics for creating a testing config file as well as explanations of the parameters to be specified. The file must be in **YAML format** and be placed in the [`configs/test`](https://github.com/Alexandre-Delplanque/phd-code/tree/main/configs/test) folder.

The config file is composed of **5 main parts**, which are detailed below:
```yaml
# 1) general settings
wandb_project: ...
wandb_entity: ...
device_name: ...
# 2) model
model: ...
# 3) dataset
dataset: ...
# 4) evaluator
evaluator: ...
# 5) stitcher
stitcher: ...
```


---

## General settings
First, you need to specify your Weights & Biaises project and entity, and the device you want to used for testing your model:
```yaml
wandb_project: 'my_project'
wandb_entity: 'my_name'
device_name: 'cuda'
```
If the project does not exist yet, it will be automatically created in your account, in the specified entity.

## Model
Then, you need to define your model and its trained parameters (stored in a `.pth` file) that will be used for testing. You also have to mention the name of your `nn.Module` class (previously defined in `animaloc.models`).

```yaml
model: 
  name: 'MyModel'      # nn.Module class name
  pth_file: null       # path to your trained parameters (.pth)
  kwargs:              # model's keyword arguments
    num_layers: 34
    img_size: [512,512]
    pretrained: True
    down_ratio: 2
    head_conv: 256
```

## Dataset
The dataset used for testing the model is defined mainly by some basic parameters, a CSV file containing the annotations (`csv_file`), a root directory containing the images (`root_dir`) and the mean and standard deviation of your dataset for normalization (`mean`, `std`).

```yaml
dataset:
  # basic parameters
  img_size: [512,512]
  anno_type: 'point'
  num_classes: 4

  name: 'CSVDataset'
  csv_file: '/your/path/to/test.csv'
  root_dir: '/your/path/to/test/images'

  mean: [0.485, 0.456, 0.406] 
  std: [0.229, 0.224, 0.225] 
```

The **`name`** key is used to specify the dataset you want to use. The available datasets can be found [here](https://github.com/Alexandre-Delplanque/phd-code/tree/main/animaloc/datasets). If you want to add a personal dataset, just create the module, add it in this [folder](https://github.com/Alexandre-Delplanque/phd-code/tree/main/animaloc/datasets) (be careful to add it also in [`__init__.py`](https://github.com/Alexandre-Delplanque/phd-code/blob/main/animaloc/datasets/__init__.py)). The only constraint is that it must take as input a CSV file as well as the other parameters listed at the beginning of this section. For convenience, it is strongly advised to inherit from the [`CSVDataset`](https://github.com/Alexandre-Delplanque/phd-code/blob/5a87a420bf6da3116796e69a84260b8dd414d54d/animaloc/datasets/csv.py#L40) class, and overwrite the `_load_image()` and `_load_target()` methods.

## Evaluator
You can specify a custom evaluator that will be used during model testing:

```yaml 
evaluator:
  name: 'Evaluator'
  threshold: 5
  kwargs:
    lmds_ks: [3,3]
```

To do this, you need to specify some parameters:

* **`name`**, class name for the evaluator to be used.
* **`threshold`**, threshold (float) used to define a True Positive (e.g. a radius value in case of point model evaluation),
* **`kwargs`**, evaluator's keyword arguments.

## Stitcher (optional)
When specified, used to perform inference on large images. The module works as a moving window on the large image. It collects the predictions of each patch and aggregates them to get the detections in the coordinate system of the whole image. For more information, check the [`Stitcher`](https://github.com/Alexandre-Delplanque/phd-code/blob/5a87a420bf6da3116796e69a84260b8dd414d54d/animaloc/eval/stitching.py#L14) class.

```yaml 
stitcher:
  name: 'Stitcher'
  kwargs:
    overlap: 100
    down_ratio: 2
    up: False
    reduction: 'mean'
```

The **`name`** key is used to specify the dataset you want to use. The available stitchers can be found [here](https://github.com/Alexandre-Delplanque/phd-code/blob/main/animaloc/eval/stitching.py). If you want to add a personal stitcher, just create the module, add it in this [file](https://github.com/Alexandre-Delplanque/phd-code/blob/main/animaloc/eval/stitching.py). For convenience, it is strongly advised to inherit from the [`Stitcher`](https://github.com/Alexandre-Delplanque/phd-code/blob/main/animaloc/eval/stitching.py#L17) class.

To not use it, specify:

```yaml 
stitcher: null
```