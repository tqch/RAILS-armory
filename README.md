# External Repo of Armory Testbed for RAILS Evaluation
(no system-level docker installation needed)

## Attacks for testing
### White-box
- FGSM
- PGD
### Black-box
- Square

## Usage

Evaluate white-box attack (FGSM) on RAILS via Armory:
```buildoutcfg
armory run example_scenario_configs/test_rails_fgsm.json --check --no-docker --use-gpu
```
note that the `--use-gpu` is optional (remove it if no gpu is installed on your machine)

Evaluate black-box attack (HopSkipJump) on RAILS via Armory:
```buildoutcfg
armory run example_scenario_configs/test_rails_hopskipjump.json --check --no-docker --use-gpu --skip-benign
```
