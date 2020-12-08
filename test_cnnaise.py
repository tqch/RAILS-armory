from mnist_aise import CNNAISE
import json,pickle
import torch
from utils.logger import get_default_logger
import time


def smoke_test():
    logger = get_default_logger("smoke_test")
    try:
        with open("smoke_test_config.json","r") as f:
            default_config = json.load(f)
    except FileNotFoundError:
        logger.warning("Weight file is not found in the path given!")
    logger.info("RAIL system starts...")
    cnnaise = CNNAISE(**default_config)
    try:
        with open("adversarial_examples/adv_20.pkl","rb") as f:
            x_adv = torch.Tensor(pickle.load(f))
    except FileNotFoundError:
        logger.warning("Pre-computed adversarial examples are not found in the path given!")
    pred = cnnaise.predict(x_adv)
    print("Predictions: {}".format(pred.argmax(axis=-1)))


if __name__ == "__main__":
    start_time = time.perf_counter()
    smoke_test()
    end_time = time.perf_counter()
    print("Elapsed time: {}s".format(end_time-start_time))
