from mnist_aise import CNNAISE
import json,pickle
import torch
with open("aise_default_config.json","r") as f:
    default_config = json.load(f)
cnnaise = CNNAISE(**default_config)
with open("examples/adv_20.pkl","rb") as f:
    x_adv = torch.Tensor(pickle.load(f))
cnnaise.predict(x_adv)
