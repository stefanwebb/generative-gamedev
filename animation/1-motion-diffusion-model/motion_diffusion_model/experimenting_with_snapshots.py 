from model import MotionDiffusionModel
import pickle
from safetensors.torch import load_file
import torch

import os

print(os.getcwd())

with open("../motion-diffusion-model/vars.pkl", "rb") as f:
    args = pickle.load(f)

model = MotionDiffusionModel()
model.load_state_dict(
    load_file(
        "./animation/1-motion-diffusion-model/motion_diffusion_model/model.safetensors"
    )
)
model.cuda()
model.eval()

output = model(args["x"].cuda(), args["timesteps"].cuda(), args["y"])

print(torch.max(output - args["output"]))
