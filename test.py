import models
import torch
import yaml
import random
import matplotlib.pyplot as plt
import matplotlib
import utils
import numpy as np
import os

from MRIDatasets import CombinedDatasetT2ABS
matplotlib.use("Agg")

def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img

def cal_content_loss(out_con, in_con):
    loss = torch.nn.MSELoss()
    return loss(out_con, in_con)

def cal_style_loss(out_feat, in_feat):
    out_feat_change = out_feat.reshape(out_feat.shape[0], out_feat.shape[1], -1)
    out_feat_trans = out_feat_change.transpose(1, 2)
    out_gram = torch.bmm(out_feat_change, out_feat_trans) / (out_feat.shape[1] * out_feat.shape[2] * out_feat.shape[3])

    in_feat_change = in_feat.reshape(in_feat.shape[0], in_feat.shape[1], -1)
    in_feat_trans = in_feat_change.transpose(1, 2)
    in_gram = torch.bmm(in_feat_change, in_feat_trans)  / (in_feat.shape[1] * in_feat.shape[2] * in_feat.shape[3])

    gram_diff = out_gram - in_gram

    frobenius = torch.norm(gram_diff, p='fro')
    return frobenius**2



Device = torch.device("cuda:0")


config_path = './config/T2W_Na.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


model = models.arrfr.ArRFR(
config=config,
encoder_spec1=config["encoder_spec"],
encoder_spec2=config["encoder_spec"],
imnet_spec=config["imnet_spec"],
local_ensemble=True,
feat_unfold=True,
cell_decode=True
).to(Device)

checkpoint = torch.load("./checkpoints/pretrained/ArRFR_sna=0_st2=10_fna=6_ft2=4.pth")

model.load_state_dict(checkpoint)

vgg = models.vgg.Vgg16().to(Device)
vgg.load_state_dict(torch.load('./checkpoints/pretrained/vgg.pth'))

test_data = CombinedDatasetT2ABS(config, mode='test')
torch.random.manual_seed(6)
random_sample = random.choice(test_data)

na = random_sample['Na']
t2 = random_sample['T2']


input_H = torch.from_numpy(t2).unsqueeze(0).float().to(Device)
input_Na = torch.from_numpy(na).unsqueeze(0).float().to(Device)
with torch.no_grad():
    model.eval()
    coords = utils.make_coord((112, 112)).unsqueeze(0).float().to(Device)
    cell_inp = torch.ones_like(coords) * 2 / 112
    output1 = model(input_Na, input_H, coords, cell_inp)
    
with torch.no_grad():
    model.eval()
    coords = utils.make_coord((224, 224)).unsqueeze(0).float().to(Device)
    cell_inp = torch.ones_like(coords) * 2 / 224
    output2 = model(input_Na, input_H, coords, cell_inp)

with torch.no_grad():
    model.eval()
    coords = utils.make_coord((224, 224)).unsqueeze(0).float().to(Device)
    cell_inp = torch.ones_like(coords) * 2 / 224
    output2 = model(input_Na, input_H, coords, cell_inp)

with torch.no_grad():
    model.eval()
    coords = utils.make_coord((280, 280)).unsqueeze(0).float().to(Device)
    cell_inp = torch.ones_like(coords) * 2 / 280
    output25 = model(input_Na, input_H, coords, cell_inp)

with torch.no_grad():
    model.eval()
    coords = utils.make_coord((336, 336)).unsqueeze(0).float().to(Device)
    cell_inp = torch.ones_like(coords) * 2 / 336
    output3 = model(input_Na, input_H, coords, cell_inp)

with torch.no_grad():
    model.eval()
    coords = utils.make_coord((392, 392)).unsqueeze(0).float().to(Device)
    cell_inp = torch.ones_like(coords) * 2 / 392
    output35 = model(input_Na, input_H, coords, cell_inp)

with torch.no_grad():
    model.eval()
    coords = utils.make_coord((448, 448)).unsqueeze(0).float().to(Device)
    cell_inp = torch.ones_like(coords) * 2 / 448
    output4 = model(input_Na, input_H, coords, cell_inp)

os.makedirs('./results', exist_ok=True) 
plt.imsave('./results/output.png',output1.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
plt.imsave('./results/outputx2.png',output2.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
plt.imsave('./results/outputx2.5.png',output25.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
plt.imsave('./results/outputx3.png',output3.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
plt.imsave('./results/outputx3.5.png',output35.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
plt.imsave('./results/outputx4.png',output4.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')


plt.imsave('./results/t2.png', np.squeeze(t2), cmap='gray')
plt.imsave('./results/na.png', np.squeeze(na), cmap='gray')

