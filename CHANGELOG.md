# [v0.2.1](https://github.com/Alexandre-Delplanque/HerdNet/releases/tag/v0.2.1) (March 21, 2024)
Code license changed to [`MIT License`](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/LICENSE.md).

## Minor Fixes
- Change UAV dataset hosting (now in [ULiège Open Data Repository](https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5))

## Commits
Alexandre-Delplanque (8):
- [c101dca](https://github.com/Alexandre-Delplanque/HerdNet/commit/c101dcafc55385be6f825edebe4834b2cad72597) - fix: update code version
- [031cc1a](https://github.com/Alexandre-Delplanque/HerdNet/commit/031cc1a1528d9a88ff4fe467ee684f22e56d93d3) - chore: minor change
- [4a2c68a](https://github.com/Alexandre-Delplanque/HerdNet/commit/4a2c68a2a49efa89e890f7ae56d3288789e22cda) - fix: update code version
- [b424c5b](https://github.com/Alexandre-Delplanque/HerdNet/commit/b424c5b7695198f179f9686d1f4dbb34015a5bbc) - Merge pull request #3 from Alexandre-Delplanque/new-license
- [ce16958](https://github.com/Alexandre-Delplanque/HerdNet/commit/ce1695888409b2e3ad4b6d244b269d81e7ffd4ee) - chore: update CHANGELOG.md
- [d2b9078](https://github.com/Alexandre-Delplanque/HerdNet/commit/d2b90785fafba495d7a0831efb823cfd4004c69d) - chore: final licence changes
- [f063a82](https://github.com/Alexandre-Delplanque/HerdNet/commit/f063a82ff4dedf0b1d748a78eb11dfde99b1cfac) - chore: switch to MIT License
- [d86ec37](https://github.com/Alexandre-Delplanque/HerdNet/commit/d86ec3736efaabbea6c1d07549e2a605a5c928ae) - fix: new link for UAV dataset #2


# [v0.2.0](https://github.com/Alexandre-Delplanque/HerdNet/releases/tag/v0.2.0) (March 29, 2023)
## New features
### Classes and functions
- `CustomLogger` : Argument to disable logging to CSV files (use to much memory).
- `Trainer`: Arguments to set the validation frequency during training (`valid_freq`) and to choose whether to save logs to CSV files (`csv_logger`).
- `HerdNet`: New method for reshaping classes (`reshape_classes()`), useful for loading pre-trained parameters.
- `FolderDataset`: New flag (`from_folder`) in `self.data` attribute.
### Python modules
- `sampler.py`: New python module for hosting samplers for data loading.
### Tools
- `train.py`: New keys: `wandb_run`, `model.freeze` (HerdNet only), `datasets.class_def`, `datasets.sampler` and `training_settings.valid_freq`. Now use the class definition (i.e., `datasets.class_def`) to make sure the labels match the species names.
- `test.py`: New keys: `wandb_run` and `dataset.class_def`. Now use the class definition (i.e., `dataset.class_def`) 1) to make sure the labels match the species names, and 2) for plotting precision-recall curves, saving the detections, the metrics and the confusion matrix.

## Commits
Alexandre-Delplanque (17):
- [ff94a5e](https://github.com/Alexandre-Delplanque/HerdNet/commit/ff94a5e4bcbe5c7efa37354f64d56859b1f0a388) - chore: update README.md
- [03ccd66](https://github.com/Alexandre-Delplanque/HerdNet/commit/03ccd66dfdd70e61677537e5755579eb4386fac6) - version: update version number and modified date
- [4791624](https://github.com/Alexandre-Delplanque/HerdNet/commit/4791624c0d32760e424d6b7b863763e54c273313) - chore: add CHANGELOG.md
- [eb159c7](https://github.com/Alexandre-Delplanque/HerdNet/commit/eb159c72270302265295a8bbff6fc85fc9b6c7a4) - docs: create doc folder and update configs md
- [f18f9f9](https://github.com/Alexandre-Delplanque/HerdNet/commit/f18f9f9ef2a8702c467d77a127333f1c7d303d60) - feat: add new keys to configs
- [7f91167](https://github.com/Alexandre-Delplanque/HerdNet/commit/7f911677237ef35aadb4378a8ef21d088a9070f8) - feat: add sampler option in train.py tool
- [bea0fc4](https://github.com/Alexandre-Delplanque/HerdNet/commit/bea0fc4cfc616c94ee0f124b635b2f3cccb250e6) - feat: samplers.py - hosts samplers for dataloading
- [d883b49](https://github.com/Alexandre-Delplanque/HerdNet/commit/d883b49bbbe954df6eb4b5c307b943a261fbb4bc) - feat: save classes, mean, std in PTH files
- [8189dc7](https://github.com/Alexandre-Delplanque/HerdNet/commit/8189dc746577375322d4d30bd6958c74a801fc6a) - feat: add 'from_folder' flag in FolderDataset data
- [907a221](https://github.com/Alexandre-Delplanque/HerdNet/commit/907a221668641627013f13831e09296c95fbeea1) - feat: +class def., +labeled results, +conf. matrix
- [7b4bc74](https://github.com/Alexandre-Delplanque/HerdNet/commit/7b4bc7461619aa3e07f474fa15e447193129dcc4) - feat: +class def., +validation freq., -cross-val
- [884673d](https://github.com/Alexandre-Delplanque/HerdNet/commit/884673d15869ecb0f8dc335f527b5a09ce0c91e4) - fix: add head_conv attribute to HerdNet module
- [7b88af8](https://github.com/Alexandre-Delplanque/HerdNet/commit/7b88af8cfeee172d7da56196727a3f6c7f8da9fb) - feat: valid_freq and csv_logger args (Trainer)
- [d88ff21](https://github.com/Alexandre-Delplanque/HerdNet/commit/d88ff21b8dab870bf6a0c04e38269ed3ee4ffc36) - feat: add option to disable csv logs
- [7071569](https://github.com/Alexandre-Delplanque/HerdNet/commit/707156928508923d9ac78efcd52ade8fe963929d) - fix: PointsToMask one-hot encoding option
- [782d877](https://github.com/Alexandre-Delplanque/HerdNet/commit/782d877fdd1a50f3590a5138814f571ded9bef26) - fix: FocalLoss, avoid NaN when output is 0 or 1
- [deec190](https://github.com/Alexandre-Delplanque/HerdNet/commit/deec19098bcbadaf73e3ff624eafa5382f5e5fec) - add article reference


# [v0.1.0](https://github.com/Alexandre-Delplanque/HerdNet/releases/tag/v0.1.0) (January 23, 2023)
Initial version of the code, used for producing the results of the reference paper "[From Crowd to Herd Counting: How to Precisely Detect and Count African Mammals using Aerial Imagery and Deep Learning?](https://doi.org/10.1016/j.isprsjprs.2023.01.025)".

## Commits
Alexandre-Delplanque (11):
- [f6586a1](https://github.com/Alexandre-Delplanque/HerdNet/commit/f6586a1bf846cf6ac66762ad84091e3476e1e435) - Add LICENSE
- [9164bd9](https://github.com/Alexandre-Delplanque/HerdNet/commit/9164bd941245701fcd1e734992d88a8edd4b9568) - Create LICENSE.md
- [e5d5e5c](https://github.com/Alexandre-Delplanque/HerdNet/commit/e5d5e5cc3589153470c4da07c2b21dd9d1415fdc) - Merge branch 'main' of https://github.com/Alexandre-Delplanque/Herd-Net
- [303ca64](https://github.com/Alexandre-Delplanque/HerdNet/commit/303ca64f2ef9fc243ff32b807cb329c36431e03d) - Upgrade infer tool and use tqdm for progress bars
- [ca8e03d](https://github.com/Alexandre-Delplanque/HerdNet/commit/ca8e03d3b0d3795e85329abf4ba7f966d6e26887) - Update Colab notebook link
- [4179b60](https://github.com/Alexandre-Delplanque/HerdNet/commit/4179b600cf9b1ac45ca4a262d0b24c952be62a99) - Add infer.py tool and demo notebook
- [48e2072](https://github.com/Alexandre-Delplanque/HerdNet/commit/48e2072bc75b696d67e7e7bcc5f7915c0ff4fe9f) - Update README.md
- [17f8efc](https://github.com/Alexandre-Delplanque/HerdNet/commit/17f8efc53ec761f9da7636d4562728e00c51072f) - Update environment.yml
- [77c3bbe](https://github.com/Alexandre-Delplanque/HerdNet/commit/77c3bbe6d3d1e17be9229c42c591f8719f0d880a) - Update README.md
- [3423d85](https://github.com/Alexandre-Delplanque/HerdNet/commit/3423d85f80a0f11c6d83d40ebe3ac9bf262d488b) - initial code commit
- [d176d9f](https://github.com/Alexandre-Delplanque/HerdNet/commit/d176d9fcc721c97888550a36b542a5bbb72f0fba) - Initial commit