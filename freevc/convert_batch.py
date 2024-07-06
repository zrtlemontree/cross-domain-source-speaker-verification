import os
import argparse
import random
import psutil
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

def builder_dir(root_dir, target_dir):
    dirName, subdirList, files = next(os.walk(root_dir))
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

def process_data_freevc(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        builder_dir(args.target, args.outdir)

    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)
    
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder(args.speaker_encoder_pt)

    print("Processing text...")
    titles, srcs, tgts = [], [], []

    lines = open(args.conversion_file, "r").read().splitlines()

    for rawline in lines:
        src, tgt = rawline.strip().split(" ")
        title = get_new_path(src, tgt)
        titles.append(title)
        srcs.append(os.path.join(args.source, src))
        tgts.append(os.path.join(args.target, tgt))

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts), total=len(titles)):
            title, src, tgt = line
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            if hps.model.use_spk:
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
            else:
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt, 
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            c = utils.get_content(cmodel, wav_src)
            
            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            write(os.path.join(args.outdir, title), hps.data.sampling_rate, audio)

    
