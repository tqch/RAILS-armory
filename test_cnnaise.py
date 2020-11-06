from mnist_aise import CNNAISE
import json,pickle
import torch
import logging
# from memory_profiler import profile

logger = logging.getLogger(__name__)


# @profile
def main():
    try:
        with open("aise_default_config.json","r") as f:
            default_config = json.load(f)
    except FileNotFoundError:
        logger.warning("Weight file is not found in the path given!")
    cnnaise = CNNAISE(**default_config)
    try:
        with open("examples/adv_20.pkl","rb") as f:
            x_adv = torch.Tensor(pickle.load(f))
    except FileNotFoundError:
        logger.warning("Pre-computed adversarial examples are not found in the path given!")
    logger.info("RAIL system starts...")
    cnnaise.predict(x_adv)


if __name__ == "__main__":
    main()
