import torch
import yaml
import os
import models
from torch.optim import lr_scheduler
import MRIDatasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
import tensorboardX

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


# load config
Device = torch.device('cuda:6')
config_path = './config/T2W_Na.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

epoch = config["epoch"]
lr = float(config["lr"])

# create model
model = models.arrfr.ArRFR(
    config=config,
    encoder_spec1=config["encoder_spec"],
    encoder_spec2=config["encoder_spec"],
    imnet_spec=config["imnet_spec"],
    local_ensemble=True,
    feat_unfold=True,
    cell_decode=True
).to(Device)

# prepare vgg and load checkpoints
vgg = models.vgg.Vgg16().to(Device)
vgg.load_state_dict(torch.load('./checkpoints/pretrained/vgg.pth'))


# prepare optimizer and scheduler
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

# prepare data
loss = torch.nn.L1Loss().to(Device)
train_dataset = MRIDatasets.CombinedDatasetT2ABS(config, mode="train", degradation=None)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

iter_loop = tqdm(range(epoch))

# prepare logs
os.makedirs(config["log_path"], exist_ok=True)
train_writer = tensorboardX.SummaryWriter(config["log_path"])


step = 0

for i in iter_loop:
    model.train()

    for sample in train_loader:
        total_loss = 0

        und_scale = np.arange(config["downsample_range"][0], config["downsample_range"][1])

        for scale in und_scale:
            h_downsampled = sample[f"T2_downsampled_{scale}"].to(Device).float()
            na_downsampled = sample[f"Na_downsampled_{scale}"].to(Device).float()
            h = sample["T2"].to(Device).float()
            na = sample["Na"].to(Device).float()
            coords = sample["coords"].to(Device).float()
            cell_inp = sample["cell_inp"].to(Device).float()
            out = model(na_downsampled, h_downsampled, coords, cell_inp)



            style_loss_t2 = cal_style_loss(vgg(out)[0], vgg(h)[0]) + \
                          cal_style_loss(vgg(out)[1], vgg(h)[1]) + \
                          cal_style_loss(vgg(out)[2], vgg(h)[2]) + \
                          cal_style_loss(vgg(out)[3], vgg(h)[3])

            feature_loss_t2 = cal_content_loss(vgg(out)[1], vgg(h)[1])

            style_loss_na = cal_style_loss(vgg(out)[0], vgg(na)[0]) + \
                          cal_style_loss(vgg(out)[1], vgg(na)[1]) + \
                          cal_style_loss(vgg(out)[2], vgg(na)[2]) + \
                          cal_style_loss(vgg(out)[3], vgg(na)[3])

            feature_loss_na = cal_content_loss(vgg(out)[1], vgg(na)[1])



            loss_val =config["feature_w_na"] * feature_loss_na + \
                      config["feature_w_t2"] * feature_loss_t2 + \
                      config["style_w_na"] * style_loss_na + \
                      config["style_w_t2"] * style_loss_t2

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            total_loss = total_loss + loss_val.item()


            del h_downsampled, na_downsampled, h, na, coords, cell_inp, out, loss_val, style_loss_t2, feature_loss_t2, style_loss_na, feature_loss_na
            gc.collect()
            torch.cuda.empty_cache()    

        avg_loss = total_loss / (und_scale.shape[0])
        iter_loop.set_postfix(loss=avg_loss)
        train_writer.add_scalar("loss", avg_loss, step+1)
        step += 1

    scheduler.step()

    if (i+1) % 10 == 0:
        os.makedirs(config["checkpoint_path"], exist_ok=True)
        checkpoint_path = os.path.join(config["checkpoint_path"], f"ArRFR_sna={config['style_w_na']}_st2={config['style_w_t2']}_fna={config['feature_w_na']}_ft2={config['feature_w_t2']}_iter{i}.pth")
        torch.save(model.state_dict(), checkpoint_path)



checkpoint_path = os.path.join(config["checkpoint_path"], f"ArRFR_sna={config['style_w_na']}_st2={config['style_w_t2']}_fna={config['feature_w_na']}_ft2={config['feature_w_t2']}.pth")
torch.save(model.state_dict(), checkpoint_path)

