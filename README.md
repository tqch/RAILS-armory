# External Repo of RAILS Evaluation via Armory Testbed
(no system-level docker installation needed)

## Attacks for testing

### White-box
- FGSM
- PGD

### Black-box
- Square

## Usage

Make integrated model ready for Armory:
```buildoutcfg
python make_saved_model.py
```

Evaluate white-box attack (PGD) on RAILS via Armory:
```buildoutcfg
armory run example_scenario_configs/MICH_cifar_rails_pgd.json --check --no-docker --use-gpu
```
note that the `--use-gpu` is optional (remove it if no gpu is installed on your machine)

Evaluate black-box attack (Square) on RAILS via Armory:
```buildoutcfg
armory run example_scenario_configs/MICH_cifar_rails_square.json --check --no-docker --use-gpu --skip-benign
```
