from mnist_aise import CNNAISE
import json,pickle
import torch
from memory_profiler import profile

@profile
def main():
    with open("aise_default_config.json","r") as f:
        default_config = json.load(f)
    cnnaise = CNNAISE(**default_config)
    with open("examples/adv_20.pkl","rb") as f:
        x_adv = torch.Tensor(pickle.load(f))
    cnnaise.predict(x_adv)

if __name__ == "__main__":
    main()