import torch
import hydra
import wandb
import animaloc
import os
import torchvision

import albumentations as A

from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Callable

from animaloc.data.transforms import DownSample
from animaloc.models.utils import load_model, LossWrapper
from animaloc.eval import Evaluator, Metrics, PointsMetrics, BoxesMetrics
from animaloc.eval.stitchers import Stitcher
from animaloc.utils.useful_funcs import mkdir
from animaloc.vizual import PlotPrecisionRecall


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
    
    model = model(**kwargs, num_classes=cfg.dataset.num_classes)
    model = LossWrapper(model, [])
    model = load_model(model, cfg.model.pth_file)
    return model

def _get_collate_fn(cfg: DictConfig) -> Callable:
    fn = cfg.dataset.collate_fn
    if fn is not None:
        fn = animaloc.data.batch_utils.__dict__[fn]
    return fn

def _define_stitcher(model: torch.nn.Module, cfg: DictConfig) -> Stitcher:

    name = cfg.stitcher.name

    assert name in animaloc.eval.stitchers.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the stitchers list'

    kwargs = dict(cfg.stitcher.kwargs)
    for k in ['model','size','device_name']:
        kwargs.pop(k, None)

    stitcher = animaloc.eval.stitchers.__dict__[name](
        model = model,
        size = cfg.dataset.img_size,
        **kwargs,
        device_name = cfg.device_name
        ) 

    return stitcher

def _define_evaluator(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    metrics: Metrics, 
    cfg: DictConfig
    ) -> Evaluator:

    name = cfg.evaluator.name

    assert name in animaloc.eval.evaluators.__dict__.keys(), \
        f'\'{name}\' class unfound, make sure you have included the class in the evaluators list'

    stitcher = None
    if cfg.stitcher is not None:
        stitcher = _define_stitcher(model, cfg)
    
    kwargs = dict(cfg.evaluator.kwargs)
    for k in ['model','dataloader','metrics','device_name','stitcher','header']:
        kwargs.pop(k, None)

    evaluator = animaloc.eval.evaluators.__dict__[name](
        model = model,
        dataloader = dataloader,
        metrics = metrics,
        device_name = cfg.device_name,
        stitcher = stitcher,
        header = '[TEST]',
        **kwargs
    )

    return evaluator


@hydra.main(config_path='../configs', config_name="config")
def main(cfg: DictConfig) -> None:

    cfg = cfg.test

    down_ratio = 1
    if 'down_ratio' in cfg.model.kwargs.keys():
        down_ratio = cfg.model.kwargs.down_ratio

    # Set up wandb
    wandb.init(
        project = cfg.wandb_project,
        entity = cfg.wandb_entity,
        config = dict(
            model = cfg.model,
            down_ratio = down_ratio,
            num_classes = cfg.dataset.num_classes,
            threshold = cfg.evaluator.threshold
            )
        )

    device = torch.device(cfg.device_name)

    # Prepare dataset and dataloader
    print('Building the test dataset ...')
    test_dataset = animaloc.datasets.__dict__[cfg.dataset.name](
        csv_file = cfg.dataset.csv_file,
        root_dir = cfg.dataset.root_dir,
        albu_transforms = [A.Normalize(cfg.dataset.mean, cfg.dataset.std)],
        end_transforms = [DownSample(down_ratio=down_ratio, anno_type=cfg.dataset.anno_type)]
        )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
        sampler=torch.utils.data.SequentialSampler(test_dataset), collate_fn=_get_collate_fn(cfg))
    
    # Build the trained model
    print('Building the trained model ...')
    model = _build_model(cfg).to(device)

    # Build the evaluator
    print('Preparing for testing ...')
    anno_type = cfg.dataset.anno_type
    if anno_type == 'point':
        metrics = PointsMetrics(radius = cfg.evaluator.threshold, num_classes = cfg.dataset.num_classes)
    elif anno_type == 'bbox':
        metrics = BoxesMetrics(iou = cfg.evaluator.threshold, num_classes = cfg.dataset.num_classes)
    else:
        raise NotImplementedError

    evaluator = _define_evaluator(model, test_dataloader, metrics, cfg)

    # Start testing
    print('Starting testing ...')
    out = evaluator.evaluate(wandb_flag=True, viz=False)

    # Save results
    print('Saving the results ...')
    plots_path = os.path.join(os.getcwd(), 'plots')
    mkdir(plots_path)
    pr_curve = PlotPrecisionRecall()
    metrics = evaluator._stored_metrics
    for c in range(1, metrics.num_classes):
        rec, pre = metrics.rec_pre_lists(c)
        pr_curve.feed(rec, pre, str(c))
    
    pr_curve.save(os.path.join(plots_path, 'precision_recall_curve.png'))
    
    res = evaluator.results
    print(res)
    # print(evaluator._stored_metrics._total_count)
    res.to_csv(os.path.join(os.getcwd(), 'metrics_results.csv'), index=False)

    evaluator.detections.to_csv(os.path.join(os.getcwd(), 'detections.csv'), index=False)

if __name__ == '__main__':
    main()