
import random
import psutil
import torch
import numpy as np


import soundfile as sf
import tqdm

from model_encoder import Encoder, Encoder_lf0
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk
import os

import subprocess
from spectrogram import logmelspectrogram
import kaldiio

import resampy
import pyworld as pw

def extract_logmel(wav_path, mean, std, sr=16000):

    wav, fs = sf.read(wav_path)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr

    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    mel = logmelspectrogram(
                x=wav,
                fs=fs,
                n_mels=80,
                n_fft=400,
                n_shift=160,
                win_length=400,
                window='hann',
                fmin=80,
                fmax=7600,
            )
    
    mel = (mel - mean) / (std + 1e-8)
    tlen = mel.shape[0]
    frame_period = 160/fs*1000
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return mel, lf0

def builder_dir(root_dir, target_dir):
    dirName, subdirList, _ = next(os.walk(root_dir))
    for subdir in sorted(subdirList):
        if not os.path.exists(os.path.join(target_dir, subdir)):
            os.makedirs(os.path.join(target_dir, subdir))
        builder_dir(os.path.join(dirName, subdir), os.path.join(target_dir, subdir))

def get_new_path(source_utt, target_utt):
    source_basename = os.path.basename(source_utt).split('.')[0]
    target_basename = os.path.basename(target_utt).split('.')[0]
    new_basename = f'{source_basename}_to_{target_basename}.wav'
    new_path = target_utt.replace(target_basename+".wav", new_basename)
    return new_path

def convert(args):
    # Load model
    print("Loading model...")
    out_dir = args.temp_dir
    os.makedirs(out_dir, exist_ok=True)
    device = args.device

    encoder = Encoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
    encoder_lf0 = Encoder_lf0()
    encoder_spk = Encoder_spk()
    decoder = Decoder_ac(dim_neck=64)
    encoder.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    decoder.to(device)

    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    encoder_spk.eval()
    decoder.eval()
    
    mel_stats = np.load('./vqmivc/mel_stats/stats.npy')
    mean = mel_stats[0]
    std = mel_stats[1]
    feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir)+'/feats.1'))

    # load the conversion file and processing
    print("Processing...")
    lines = open(args.conversion_file, "r").read().splitlines()

    for line in tqdm.tqdm(lines, total=len(lines)):
        src_wav_path, ref_wav_path = line.split(" ")
        out_filename = get_new_path(src_wav_path, ref_wav_path)
        # print(os.path.join(args.output, out_filename))

        out_filename = os.path.join(args.output, out_filename).split(".")[0]
        src_wav_path = os.path.join(args.source, src_wav_path)
        ref_wav_path = os.path.join(args.target, ref_wav_path)
        # print(src_wav_path, ref_wav_path, out_filename)
        # quit()
        src_mel, src_lf0 = extract_logmel(src_wav_path, mean, std)
        ref_mel, _ = extract_logmel(ref_wav_path, mean, std)
        src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
        src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(device)
        ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z, _, _, _ = encoder.encode(src_mel)
            lf0_embs = encoder_lf0(src_lf0)
            spk_emb = encoder_spk(ref_mel)
            output = decoder(z, lf0_embs, spk_emb)
            
            feat_writer[out_filename] = output.squeeze(0).cpu().numpy()

    feat_writer.close()
    print('synthesize waveform...')
    cmd = ['parallel-wavegan-decode', '--checkpoint', \
           './vqmivc/vocoder/checkpoint-3000000steps.pkl', \
           '--feats-scp', f'{str(out_dir)}/feats.1.scp', '--outdir', str(out_dir)]
    subprocess.call(cmd)

def process_data_vqmivc(args):
    if not os.path.exists(args.output):
        print("marking dir...")
        os.makedirs(args.output)
        builder_dir(args.target, args.output)
    convert(args)
