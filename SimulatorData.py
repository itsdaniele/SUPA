import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from tqdm import tqdm

INF = math.inf

def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)

class SimulatorDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, layer=0, input_dim=3, normalize_std_per_axis=True, all_points_mean=None, all_points_std=None):

        self.normalize_std_per_axis = normalize_std_per_axis
        self.all_points_mean = all_points_mean
        self.all_points_std = all_points_std

        self.input_dim = input_dim

        # filelist = os.listdir(root_path)
        filelist = sorted([os.path.abspath(os.path.join(root_path, x)) for x in os.listdir(root_path) if 'shower' in x])

        all_pcs = []

        for shower_file in tqdm(filelist[:1]):
            showers = torch.load(shower_file)
            all_pcs.extend([x[layer] for x in showers])

        self.all_points, self.masks = self.pad(all_pcs)

        del all_pcs

        # convert energy unit
        #self.all_points[:,:,-1] =  torch.nan_to_num(torch.log(self.all_points[:,:,-1]*1000), nan=0.0, posinf=0.0, neginf=0.0) # to MeV
        self.all_points[:,:,-1] =  self.all_points[:,:,-1]*1000 # convert to MeV


        if self.input_dim < 3:
            # first self.input_dim dimensions
            self.all_points = self.all_points[:,:,:self.input_dim]

        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std

        else:  # normalize across the dataset
            all_points_flattened = self.all_points[~self.masks]
            self.all_points_mean = all_points_flattened.mean(axis=0).reshape(1, 1, input_dim)

            if normalize_std_per_axis:
                self.all_points_std = all_points_flattened.std(axis=0).reshape(1, 1, input_dim)

                self.all_points_std[self.all_points_std == 0.] = 1. 
            else:
                self.all_points_std = all_points_flattened.reshape(-1).std(axis=0).reshape(1, 1, 1)

        print('Dataset stats : ', self.get_pc_stats(0))

    def pad(self, showers, max_len=None):
        # showers : List[np.array(N, 3), ..], max_len : int

        showers_len = torch.tensor([len(shower) for shower in showers])

        if max_len is None:
            max_len = int(torch.max(showers_len))
            self.max_len = max_len

        ids = torch.arange(0, max_len, out = torch.LongTensor(max_len))

        masks = (ids >= showers_len.unsqueeze(1))

        showers_pad = torch.stack([torch.cat([torch.tensor(shower, dtype=float), torch.zeros(max_len - len(shower), len(shower[0]))]) for shower in showers if len(shower) > 0])

        return showers_pad, masks

    def save_data_tensor(self, file_path):
        path1 = os.path.join(file_path, f'gt.bin')
        torch.save(self.all_points, path1)

        path2 = os.path.join(file_path, f'gt_masks.bin')
        torch.save(self.masks, path2)

        print(f'Saved data at {path1}')


    def __len__(self):
        return len(self.all_points)

    def get_pc_stats(self, idx):
        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def __getitem__(self, item):

        set = self.all_points[item]
        m, s = self.get_pc_stats(item)
        set = (set - m) / s

        mask = self.masks[item]
        c = torch.sum(~mask)

        ret = {
            'idx': item,
            'set': set,
            'mask': mask,
            'cardinality': c,
            'mean': m,
            'std': s
        }

        return ret

def collate_fn(batch):
    ret = dict()
    for k, v in batch[0].items():
        ret.update({k: [b[k] for b in batch]})

    s = torch.stack(ret['set'], dim=0).float()  # [B, N, 2]
    mask = torch.stack(ret['mask'], dim=0).bool()  # [B, N]
    cardinality = (~mask).long().sum(dim=-1)  # [B,]
    mean = torch.stack(ret['mean'], dim=0).float()
    std = torch.stack(ret['std'], dim=0).float()
    offset = torch.stack(ret['offset'], dim=0).float()

    ret.update({'set': s, 'offset':offset, 'set_mask': mask, 'cardinality': cardinality,
                'mean': mean, 'std': std})
    return ret


def build(args):
    full_dataset = SimulatorDataset(root_path=args.root_path,
                               layer=args.layer,
                               input_dim=args.input_dim,
                               normalize_std_per_axis=args.normalize_std_per_axis,
                               all_points_mean=None,
                               all_points_std=None
                               )
    max_len = full_dataset.max_len

    train_data_size = int(len(full_dataset)*args.train_val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_data_size, len(full_dataset) - train_data_size], generator=torch.Generator().manual_seed(42))

    train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              pin_memory=True, sampler=train_sampler, drop_last=True, 
                              collate_fn=collate_fn, worker_init_fn=init_np_seed) # num_workers=args.num_workers,

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, drop_last=True,
                            collate_fn=collate_fn, worker_init_fn=init_np_seed) # num_workers=args.num_workers,

    return train_dataset, val_dataset, train_loader, val_loader, max_len
