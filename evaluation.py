import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from dataset import Vox256_eval, Taichi_eval, TED_eval, Macron
from torch.utils import data
import lpips


class Eva(nn.Module):
    def __init__(self, args):
        super(Eva, self).__init__()

        self.args = args

        transform = torchvision.transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )

        if args.dataset == 'vox':
            path = 'checkpoints/vox.pt'
            dataset = Vox256_eval(transform)
        elif args.dataset == 'taichi':
            path = 'checkpoints/taichi.pt'
            dataset = Taichi_eval(transform)
        elif args.dataset == 'ted':
            path = 'checkpoints/ted.pt'
            dataset = TED_eval(transform)
        elif args.dataset == 'macron':
            path = 'exps/v1/checkpoint/806000.pt'
            # path = 'checkpoints/vox.pt'
            macron_rootdir = '/mnt/ddr/data/Macron/'
            dataset = Macron(macron_rootdir, 'test', transform)
        else:
            raise NotImplementedError

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')
        self.loader = data.DataLoader(
            dataset,
            num_workers=1,
            batch_size=1,
            drop_last=False,
        )

        self.loss_fn = lpips.LPIPS(net='alex').cuda()

    def run(self):

        loss_list = []
        loss_lpips = []
        for _ in range(10):  # repeat random loader
            for img_source, img_target in tqdm(self.loader):

                with torch.no_grad():
                    img_target = img_target.cuda()
                    img_source = img_source.cuda()
                    img_recon = self.gen(img_source, img_target)

                    loss_list.append(torch.abs(0.5 * (img_recon.clamp(-1, 1) - img_target)).mean().cpu().numpy())
                    loss_lpips.append(self.loss_fn(img_target, img_recon.clamp(-1, 1)).mean().cpu().detach().numpy())

        print("reconstruction loss: %s" % np.mean(loss_list))
        print("lpips loss: %s" % np.mean(loss_lpips))


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel_multiplier", type=int, default=1, help=' %(default)s')
    parser.add_argument("--size", type=int, default=256, help=' %(default)s')
    parser.add_argument("--latent_dim_style", type=int, default=512, help=' %(default)s')
    parser.add_argument("--latent_dim_motion", type=int, default=20, help=' %(default)s')
    parser.add_argument("--dataset", type=str, choices=['vox', 'taichi', 'ted', 'macron'], required=True, help=' %(default)s')
    args = parser.parse_args()

    demo = Eva(args)
    demo.run()
