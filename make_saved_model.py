from torchvision import datasets,transforms
import torch
from armory import paths
import os
import shutil

root = "./datasets"
download = True  # set to True for the first run
transform = transforms.ToTensor()

trainset = datasets.CIFAR10(root=root, download=download, train=True, transform=transform)

query_objects = dict()

for layer,f in [(2,256*4*4),(3,512*2*2)]:
    
    
    object_path_list = [f"class{cls}l{layer+1}.ann" for cls in range(10)]

    query_objects[str(layer)] = {
        "paths": object_path_list,
        "f": f,
        "metric": "euclidean"
    }

state_dict = torch.load("model_weights/cifar_vgg16.pt")
train_data = torch.FloatTensor(trainset.data/255).permute(0,3,1,2)
train_targets = torch.LongTensor(trainset.targets)

checkpoint = {
    "state_dict": state_dict,
    "train_data": train_data,
    "train_targets": train_targets,
    "query_objects": query_objects
}

paths.set_mode("host")  # important: set mode to host
saved_model_dir = paths.runtime_paths().saved_model_dir

shutil.copytree("./query_objects", os.path.join(saved_model_dir,"query_objects"))

torch.save(checkpoint, os.path.join(saved_model_dir,"cifar_vgg16.pt"))