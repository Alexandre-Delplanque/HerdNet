__copyright__ = \
    """
    Copyright (C) 2022 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the CC BY-NC-SA-4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/). 
    It is to be used for academic research purposes only, no commercial use is permitted.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 17, 2023
    """
__author__ = "Alexandre Delplanque"
__license__ = "CC BY-NC-SA 4.0"
__version__ = "0.1.0"


import torch
import hydra
import animaloc
import wandb
import pandas
import os
import torchvision

import albumentations as A

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from omegaconf import DictConfig
from typing import Callable, Optional

from animaloc.data.utils import group_by_image, weighted_samples
from animaloc.models.utils import LossWrapper, load_model
from animaloc.eval import Evaluator, PointsMetrics, Stitcher, BoxesMetrics, ImageLevelMetrics

from animaloc.utils.seed import set_seed
from animaloc.utils.useful_funcs import current_date, mkdir

def _set_species_labels(cls_dict: dict, df: pandas.DataFrame) -> None:
    assert 'species' in df.columns
    cls_dict = dict(map(reversed, cls_dict.items()))
    df['labels'] = df['species'].map(cls_dict)

def _load_albu_transforms(tr_cfg: dict) -> list:
    transforms = []
    for name , kwargs in tr_cfg.items():
        transforms.append(A.__dict__[name](**kwargs))
    
    return transforms

def _load_end_transforms(tr_cfg: DictConfig) -> Optional[list]:

    if tr_cfg is not None:
        transforms = []
        for name , kwargs in tr_cfg.items():
            
            if name == 'MultiTransformsWrapper':
                tr_list = []
                for n, k in kwargs.items():
                    tr_list.append(animaloc.data.transforms.__dict__[n](**k))
                
                transforms.append(animaloc.data.transforms.__dict__[name](tr_list))

            else:
                transforms.append(animaloc.data.transforms.__dict__[name](**kwargs))

        return transforms
 
    else:
        return None

def _build_sampler(df: pandas.DataFrame, col: str) -> Sampler:
    data = group_by_image(df)
    target_var = torch.tensor(data[col].values)
    samples = weighted_samples(target_var, [0.0, 1.0])
    sampler = WeightedRandomSampler(samples, len(samples), replacement=False)

    return sampler

def _get_collate_fn(cfg: DictConfig) -> Callable:
    fn = cfg.datasets.collate_fn
    if fn is not None:
        fn = animaloc.data.batch_utils.__dict__[fn]
    return fn

def _build_model(cfg: DictConfig) -> torch.nn.Module:

    name = cfg.model.name
    from_torchvision = cfg.model.from_torchvision

    if from_torchvision:
        assert name in torchvision.models.__dict__.keys(), \
            f'\'{name}\' unfound in torchvision\'s models'

        model = torchvision.models.__dict__[name]

    else:
        assert name in animaloc.models.__dict__.keys(), \
            f'\'{name}\' class unfound, make sure you have included the class in the models list'

        model = animaloc.models.__dict__[name]

    kwargs = dict(cfg.model.kwargs)
    for k in ['num_classes']:
        kwargs.pop(k, None)

    model = model(**kwargs, num_classes=cfg.datasets.num_classes)

    return model

def _load_losses(cfg: DictConfig) -> tuple:
    criterions = []
    if cfg.losses is not None:
        for loss, args in cfg.losses.items():
            
            kwargs = {}
            if 'kwargs' in args.keys():
                kwargs = dict(args.kwargs)

                if 'weights' in kwargs.keys():
                    kwargs['weights'] = torch.Tensor(kwargs['weights'])
                elif 'weight' in kwargs.keys():
                    kwargs['weight'] = torch.Tensor(kwargs['weight']).to(torch.device(cfg.device_name))

            crit_dict = {}
            if args.from_torch:
                crit_dict.update({'loss': torch.nn.__dict__[loss](**kwargs)})
            else:
                crit_dict.update({'loss': animaloc.train.losses.__dict__[loss](**kwargs)})

            crit_dict.update({
                'idx': args.output_idx, 
                'idy': args.target_idx,
                'lambda': args.lambda_const, 
                'name': args.print_name
                })

            criterions.append(crit_dict)
    
    return criterions

def _define_stitcher(
    model: torch.nn.Module,
    cfg: DictConfig
    ) -> Stitcher:

    kwargs = dict(cfg.training_settings.stitcher.kwargs)
    for k in ['model','size','device_name']:
        kwargs.pop(k, None)

    stitcher = animaloc.eval.stitchers.__dict__[cfg.training_settings.stitcher.name](
        model = model,
        size = cfg.datasets.img_size,
        **kwargs,
        device_name = cfg.device_name
        ) 

    return stitcher


def _define_evaluator(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader,  
    cfg: DictConfig
    ) -> Evaluator:

    name = cfg.training_settings.evaluator.name
    anno_type = cfg.datasets.anno_type

    assert name in animaloc.eval.evaluators.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the evaluators list'

    if anno_type == 'point':
        metrics = PointsMetrics(
            radius = cfg.training_settings.evaluator.threshold, 
            num_classes = cfg.datasets.num_classes
            )
    elif anno_type == 'bbox':
        metrics = BoxesMetrics(
            iou = cfg.training_settings.evaluator.threshold, 
            num_classes = cfg.datasets.num_classes
            )
    elif anno_type == 'image':
        metrics = ImageLevelMetrics(
            num_classes = cfg.datasets.num_classes
            )
    else:
        raise NotImplementedError

    stitcher = None
    if cfg.training_settings.stitcher is not None:
        stitcher = _define_stitcher(model, cfg)
    
    kwargs = dict(cfg.training_settings.evaluator.kwargs)
    for k in ['model','dataloader','metrics','device_name','stitcher','header', 'vizual_fn']:
        kwargs.pop(k, None)

    vizual_fn = None
    if cfg.training_settings.vizual_fn is not None:
        vizual_fn = animaloc.vizual.plots.__dict__[cfg.training_settings.vizual_fn]
        
    evaluator = animaloc.eval.evaluators.__dict__[name](
        model = model,
        dataloader = dataloader,
        metrics = metrics,
        device_name = cfg.device_name,
        stitcher = stitcher,
        header = '[TEST]',
        vizual_fn = vizual_fn, 
        **kwargs
    )

    return evaluator

@hydra.main(config_path='../configs', config_name="config")
def main(cfg: DictConfig) -> None:

    cfg = cfg.train

    # Set the seed
    print(f'Setting the seed to {cfg.seed}')
    set_seed(cfg.seed)

    # Prepare datasets and dataloaders
    print('Building datasets ...')
    device = torch.device(cfg.device_name)

    train_args = cfg.datasets.train
    val_args = cfg.datasets.validate

    train_df = pandas.read_csv(train_args.csv_file)
    _set_species_labels(dict(cfg.datasets.class_def), train_df)

    train_dataset = animaloc.datasets.__dict__[train_args.name](
        csv_file = train_df,
        root_dir = train_args.root_dir,
        albu_transforms = _load_albu_transforms(train_args.albu_transforms),
        end_transforms = _load_end_transforms(train_args.end_transforms)
        )
    
    sampler = None
    train_shuffle = True
    if train_args.sample_on is not None:
        sampler = _build_sampler(df=train_dataset.data, col=train_args.sample_on)
        train_shuffle = False

    train_dataloader = DataLoader(train_dataset, 
        batch_size=cfg.training_settings.batch_size, shuffle=train_shuffle, sampler=sampler,
        collate_fn=_get_collate_fn(cfg))
    
    val_dataloader = None
    if val_args is not None:

        val_df = pandas.read_csv(val_args.csv_file)
        _set_species_labels(dict(cfg.datasets.class_def), val_df)
        
        val_dataset = animaloc.datasets.__dict__[val_args.name](
            csv_file = val_df,
            root_dir = val_args.root_dir,
            albu_transforms = _load_albu_transforms(val_args.albu_transforms),
            end_transforms = _load_end_transforms(val_args.end_transforms)
            )
        
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=_get_collate_fn(cfg))

    work_dir = None

    # Set up wandb
    print('Connecting to Weights & Biases ...')
    settings = cfg.training_settings
    losses = cfg.losses
    if losses is not None:
        losses = list(cfg.losses.keys())

    wandb.init(
        project = cfg.wandb_project,
        entity = cfg.wandb_entity,
        config = dict(
            batch_size = settings.batch_size,
            optimizer = settings.optimizer,
            lr = settings.lr,
            weight_decay = settings.weight_decay,
            warmup_iters = settings.warmup_iters,
            epochs = settings.epochs,
            losses = losses,
            seed = cfg.seed,
            data_augmentation = list(cfg.datasets.train.albu_transforms.keys()),
            input_size = cfg.datasets.img_size,
            **cfg.model.kwargs
            )
        )
    
    date = current_date()
    wandb.run.name = f'{date}_' + cfg.wandb_run + f'_RUN_{wandb.run.id}'
    
    # Build the model
    print('Building the model ...')
    model = _build_model(cfg)

    # Prepare for training
    print('Preparing for training ...')
    criterions = _load_losses(cfg)
    model = LossWrapper(model, criterions).to(device)

    if cfg.model.load_from is not None:
        model = load_model(model, cfg.model.load_from)

        if 'HerdNet' in cfg.model.name:
            if cfg.model.freeze is not None:
                model.model.freeze(layers=list(cfg.model.freeze))
                print(f"Layers {list(cfg.model.freeze)} freezed")
    
    if cfg.training_settings.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr = cfg.training_settings.lr, 
            weight_decay = cfg.training_settings.weight_decay
            )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr = cfg.training_settings.lr, 
            weight_decay = cfg.training_settings.weight_decay
            )
    
    # Watch the model's gradients during training
    wandb.watch(model)
    
    # Evaluator ?
    evaluator = None
    validate_on = 'recall'
    select = 'min'
    if cfg.training_settings.evaluator is not None:

        assert val_dataloader is not None, \
            'A validation dataset must be defined to build an evaluator'

        evaluator = _define_evaluator(model, val_dataloader, cfg)
        select = cfg.training_settings.evaluator.select_mode
        validate_on = cfg.training_settings.evaluator.validate_on
    
    # Start training & validation
    auto_lr = cfg.training_settings.auto_lr
    if auto_lr:
        auto_lr = dict(cfg.training_settings.auto_lr)

    vizual_fn = None
    if cfg.training_settings.vizual_fn is not None:
        vizual_fn = animaloc.vizual.plots.__dict__[cfg.training_settings.vizual_fn]

    trainer = animaloc.train.trainers.__dict__[cfg.training_settings.trainer](
        model, 
        train_dataloader, 
        optimizer = optimizer, 
        num_epochs = cfg.training_settings.epochs, 
        auto_lr = auto_lr,
        # adaloss = cfg.training_settings.adaloss,
        val_dataloader = val_dataloader, 
        evaluator = evaluator,
        device_name = cfg.device_name,
        vizual_fn = vizual_fn,
        work_dir = work_dir,
        print_freq = cfg.training_settings.print_freq,
        valid_freq = cfg.training_settings.valid_freq,
        )
    
    if cfg.model.resume_from is not None:
        print(f'Resuming training from \'{cfg.model.resume_from}\' ...')
        trainer.resume(
            pth_path = cfg.model.resume_from, 
            select = select,
            validate_on = validate_on, 
            load_optim = True,
            wandb_flag = True
            )
    else:
        print('Starting training ...')
        trainer.start(
            cfg.training_settings.warmup_iters, 
            select = select,
            validate_on = validate_on, 
            wandb_flag = True
            )

if __name__ == '__main__':
    main()