import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
import random
import subprocess
import re

import tempfile

from gpu import lock_gpu


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    # img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    img = torch.from_numpy(img).float()  # [0, 1]
    img_norm = (img - 0.5) * 2.0  # [-1, 1]

    return img_norm


def vid_preprocessing(vid_path):
    size=256
    vframes, _, infodict = torchvision.io.read_video(vid_path, pts_unit='sec')  # , 256)
    vid = vframes.permute(0, 3, 1, 2)  # .unsqueeze(0)
    vid = torchvision.transforms.Resize(size)(vid)
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    fps = infodict['video_fps']

    return vid_norm, fps


def save_video(vid_target_recon, original_video_path, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 1)  # channel last

    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')
    # create temporary file
    temp_save_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    torchvision.io.write_video(temp_save_file.name, vid, int(fps))
    # print(f'temp video duration {get_video_duration(temp_save_file.name):.5f} s')
    # print(f'original video duration {get_video_duration(original_video_path):.5f} s')
    add_audio_to_video(video_path=temp_save_file.name, audio_path=original_video_path,
                       output_video_path=save_path)
    # print(f'saved video duration {get_video_duration(save_path):.5f} s')
    # delete temporary file
    temp_save_file.close()
    os.unlink(temp_save_file.name)


def add_audio_to_video(video_path, audio_path, output_video_path):
    """ Ajoute l'audio de 'video_path' à 'processed_video_path' et sauvegarde le résultat dans 'output_video_path'. """
    subprocess.run([
        "ffmpeg", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "copy", "-y", "-loglevel", "panic", "-strict", "experimental",
        "-map", "0:v:0", "-map", "1:a:0", output_video_path
    ])


def get_video_duration(filename):
    """ Retourne la durée de la vidéo en secondes. """
    result = subprocess.run(["ffmpeg", "-i", filename], stderr=subprocess.PIPE, text=True)
    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    if duration_match:
        hours, minutes, seconds = map(float, duration_match.groups())
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Could not retrieve video duration.")


def extract_first_frame(video_path, tmp_image_path, size):
    subprocess.run([
        "ffmpeg", "-i", video_path, "-y", "-loglevel", "panic",
        "-frames:v", "1", tmp_image_path
    ])
    # "-vf scale=", "%d:%d" % (size, size),
    return 0


class VideoDataset(Dataset):
    def __init__(self, vox_crop_root_dir, size):
        self.vox_crop_root_dir = vox_crop_root_dir
        self.file_list = []
        for dirpath, dirnames, filenames in os.walk(self.vox_crop_root_dir):
            for filename in filenames:
                if filename.endswith(".mp4"):
                    self.file_list.append(os.path.join(dirpath, filename))
        self.size = size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        driving_path = self.file_list[idx]

        dirpath = os.path.dirname(driving_path)
        if '/data/vox/' in dirpath:
            save_folder = dirpath.replace('/data/vox/dev_crop', '/data/vox/dev_crop_lia')
        else:
            save_folder = dirpath.replace('/vox/dev_crop', '/vox/dev_crop_lia')
        vid_output_path = os.path.join(save_folder, os.path.basename(driving_path))
        if os.path.exists(vid_output_path):
            return None, None, vid_output_path, driving_path, None

        os.makedirs(save_folder, exist_ok=True)
        tmp_image_path = os.path.join(save_folder, 'tmp.jpg')
        extract_first_frame(driving_path, tmp_image_path, self.size)

        img_source = img_preprocessing(tmp_image_path, self.size)
        vid_target, fps = vid_preprocessing(driving_path)

        return img_source, vid_target, vid_output_path, driving_path, fps

class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        if args.model == 'vox':
            model_path = 'checkpoints/vox.pt'
        elif args.model == 'taichi':
            model_path = 'checkpoints/taichi.pt'
        elif args.model == 'ted':
            model_path = 'checkpoints/ted.pt'
        elif args.model == 'macron':
            model_path = 'exps/v1/checkpoint/806000.pt'
        else:
            raise NotImplementedError

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')

    def run(self, dataset):
        batch_size = self.args.batch_size
        print('==> running')
        tic = time.time()
        count = 0
        for img_source, vid_target, vid_output_path, driving_path, fps in dataset:
            # processing a single video
            tic_local = time.time()
            print('Processing', driving_path)
            print('output path', vid_output_path)
            if img_source is None:
                print('skipping', driving_path)
                continue
            with (torch.no_grad()):

                img_source = img_source.cuda()
                # stupid to send the whole video to cuda here
                # vid_target = vid_target.cuda()
                vid_target_recon = []

                if self.args.model == 'ted':
                    h_start = None
                else:
                    h_start = self.gen.enc.enc_motion(vid_target[0, :, :, :].unsqueeze(0).cuda())

                num_frames = vid_target.size(0)
                # creating batches from video frames
                batches = [range(i*batch_size, (i+1)*batch_size) for i in range(num_frames//batch_size)]
                for i, batch_idx in enumerate(batches):
                    img_target = torch.cat([vid_target[j, :, :, :].unsqueeze(0) for j in batch_idx], dim=0)
                    img_target = img_target.cuda()
                    img_source_repeat = img_source.repeat(img_target.size(0), 1, 1, 1)
                    img_recon = self.gen(img_source_repeat, img_target, h_start)
                    vid_target_recon.append(img_recon)
                print('num_frames % batch_size = ', num_frames % batch_size)
                # last batch is not of batch_size size
                if num_frames % batch_size != 0:
                    last_processed_frame = batches[-1][-1]
                    img_target = torch.cat([vid_target[j, :, :, :].unsqueeze(0) for j in range(last_processed_frame+1,
                                                                                               num_frames)], dim=0)
                    img_target = img_target.cuda()
                    img_source_repeat = img_source.repeat(img_target.size(0), 1, 1, 1)
                    img_recon = self.gen(img_source_repeat, img_target, h_start)
                    vid_target_recon.append(img_recon)

                vid_target_recon = torch.cat(vid_target_recon, dim=0)

                save_video(vid_target_recon, driving_path, vid_output_path, fps)
            print(f'Elapsed time : {(time.time() - tic_local):.2f} s')
            count += 1
        print(f'Total elapsed time : {(time.time() - tic):.2f} s')


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--root_dir', type=str, default='/data/anasynth_nonbp/benaroya/data/vox/dev_crop')
    parser.add_argument('--cpu', help='Use CPU', action='store_true')
    args = parser.parse_args()

    args.size = 256
    args.channel_multiplier = 1
    args.model = 'vox'
    args.latent_dim_style = 512
    args.latent_dim_motion = 20

    lock_gpu(args.cpu)

    tic = time.time()
    count = 0
    dataset = VideoDataset(vox_crop_root_dir=args.root_dir, size=args.size)

    # inference
    demo = Demo(args)
    demo.run(dataset)
