# External Repo of Armory Testbed for RAILS Evaluation
(no system-level docker installation needed)

## Attacks tested
### White-box
- FGSM
- PGD
### Black-box
- ZOO
- Patch
- Shadow
- Square
- Threshold
- HopSkipJump
- Wasserstein (non $l_p$)

## Usage

Smoke test for CNN+AISE defense:
```buildoutcfg
python -m test_cnnaise
```

Evaluate white-box attack (FGSM) on RAILS via Armory:
```buildoutcfg
armory run example_scenario_configs/test_rails_fgsm.json --check --no-docker --use-gpu
```
note that the `--use-gpu` is optional (remove if no gpu is installed on your machine)

Evaluate black-box attack (HopSkipJump) on RAILS via Armory:
```buildoutcfg
armory run example_scenario_configs/test_rails_hopskipjump.json --check --no-docker --use-gpu --skip-benign
```