from torch.utils.data import Dataset
import os
import numpy as np
import utils
import torch
import monai
import yaml

config_path = './config/T2W_Na.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


def degrade_rician(x):
    noise = monai.transforms.RandRicianNoise(prob=1.0, std=0.4, relative=True, channel_wise=True, sample_std=False)
    x_degraded = noise(x)
    return x_degraded

    

class CombinedDatasetT2ABS(Dataset):
    def __init__(self, config, mode, degradation=None):
        self.config = config
        self.mode = mode
        self.degradation = degradation
        self.data_pairs = self._load_data_pairs()

    def _load_data_pairs(self):
        data_pairs = []
        for file_name_na in os.listdir(self.config["input_dir_Na"]):

            # if testing
            if self.mode == 'test':
                if file_name_na.endswith(".npy") and "Na" in file_name_na and ("ZAQ" in file_name_na or "ZJ"in file_name_na or "WJJ" in file_name_na or "ZP" in file_name_na or "SSJ" in file_name_na or "GR" in file_name_na): 
                    file_name_h = file_name_na.replace('Na', 'T2')
                    if file_name_h in os.listdir(self.config["input_dir_H"]):
                        data_pairs.append((file_name_na, file_name_h))
            # if training
            else:
                if file_name_na.endswith("npy") and "Na" in file_name_na and "ZAQ" not in file_name_na and "ZJ" not in file_name_na and "WJJ" not in file_name_na and "ZP" not in file_name_na and "SSJ" not in file_name_na and "GR" not in file_name_na:
                    file_name_h = file_name_na.replace('Na', 'T2')
                    if file_name_h in os.listdir(self.config["input_dir_H"]):
                        data_pairs.append((file_name_na, file_name_h))

        return data_pairs



    def __len__(self):
        return len(self.data_pairs)
      
    def __getitem__(self, idx):
        na_file, h_file = self.data_pairs[idx]
        data_Na = np.load(os.path.join(self.config["input_dir_Na"], na_file)) # (112, 112, 2)
        data_T2 = np.load(os.path.join(self.config["input_dir_H"], h_file)) # (112, 112)

        data_Na = np.transpose(data_Na, (2, 0, 1))  # (2, 112, 112)

        data_Na_cplx = data_Na[0] + 1j * data_Na[1]
        data_Na_abs = np.abs(data_Na_cplx) # (112, 112)

        data_T2 = np.expand_dims(data_T2, axis=0) # (1, 112, 112)
        data_Na_abs = np.expand_dims(data_Na_abs, axis=0) # (1, 112, 112)


        # normalize

        data_Na_abs_normalized = (data_Na_abs - np.min(data_Na_abs)) / (np.max(data_Na_abs) - np.min(data_Na_abs)) 
        data_T2_normalized = (data_T2 - np.min(data_T2)) / (np.max(data_T2) - np.min(data_T2)) 



        

        # # make coordinates and cell for cell decoding
        coords = utils.make_coord((data_Na.shape[1], data_Na.shape[2])).float() # shape: [H*W, 2] 
        cell_inp = torch.ones_like(coords) * 2 / data_Na.shape[1]
        # # create a sample
        sample = {
            "Na": data_Na_abs_normalized,
            "T2": data_T2_normalized,
            "coords": coords,
            "cell_inp": cell_inp,
        }

        # # downsample the image with different scales
        downsample_scale = np.arange(self.config["downsample_range"][0], self.config["downsample_range"][1])


        for scale in downsample_scale:
            data_T2_downsampled = data_T2_normalized[:, ::scale, ::scale]
            sample[f"T2_downsampled_{scale}"] = data_T2_downsampled

        if self.degradation == None:
            for scale in downsample_scale:  
                data_Na_downsampled = data_Na_abs_normalized[:, ::scale, ::scale]
                sample[f"Na_downsampled_{scale}"] = data_Na_downsampled

                
        if self.degradation == 'rician':
            data_Na_abs_degraded = degrade_rician(data_Na_abs) # (1, 112, 112)
            data_Na_abs_degraded_normalized = (data_Na_abs_degraded - np.min(data_Na_abs_degraded)) / (np.max(data_Na_abs_degraded) - np.min(data_Na_abs_degraded)) # (1, 112, 112)
            for scale in downsample_scale:  
                data_Na_downsampled = data_Na_abs_degraded_normalized[:, ::scale, ::scale]
                sample[f"Na_downsampled_{scale}"] = data_Na_downsampled

                
        return sample

