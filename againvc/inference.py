import os
import torch
import yaml
from util.parser import get_parser
from util.config import Config
from util.mytorch import same_seeds
from agent.inferencer import Inferencer
import logging
from tools import HParams
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s | %(filename)s | %(message)s',\
     datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def get_args():
    parser = get_parser(description='Inference')

    # required
    parser.add_argument('--load', '-l', type=str, help='Load a checkpoint.', required=True)
    parser.add_argument('--source', '-s', help='Source path. A .wav file or a directory containing .wav files.', required=True)
    parser.add_argument('--target', '-t', help='Target path. A .wav file or a directory containing .wav files.', required=True)
    parser.add_argument('--conversion_file', '-conv', help='The file of conversion list', required=True)
    parser.add_argument('--output', '-o', help='Output directory.', required=True)
    parser.add_argument("--device", type=str, default="cuda:0", help="device")

    # config
    parser.add_argument('--config', '-c', help='The train config with respect to the model resumed.', default='./config/train.yaml')
    parser.add_argument('--dsp-config', '-d', help='The dsp config with respect to the training data.', default='./config/preprocess.yaml')
   
    # dryrun
    parser.add_argument('--dry', action='store_true', help='whether to dry run')
    # debugging mode
    parser.add_argument('--debug', action='store_true', help='debugging mode')

    # seed
    parser.add_argument('--seed', type=int, help='random seed', default=961998)

    # [--log-steps <LOG_STEPS>]
    parser.add_argument('--njobs', '-p', type=int, help='', default=4)
    parser.add_argument('--seglen', help='Segment length.', type=int, default=None)

    return parser.parse_args()


def get_config(file):
    with open(file, "r", encoding="utf-8") as f:
        result = f.read()
        result = yaml.load(result, Loader=yaml.FullLoader)
    args = HParams(**result)
    return args

def builder_dir(root_dir, target_dir):
    dirName, subdirList, files = next(os.walk(root_dir))
    for subdir in sorted(subdirList):
        if not os.path.exists(os.path.join(target_dir, subdir)):
            os.makedirs(os.path.join(target_dir, subdir))
        builder_dir(os.path.join(dirName, subdir), os.path.join(target_dir, subdir))


if __name__ == '__main__':
    # config
    args = get_args()
    config = Config(args.config)
    same_seeds(args.seed)
    args.dsp_config = Config(args.dsp_config)

    # log some info
    logger.info(f'Config file:  {args.config}')
    logger.info(f'Checkpoint:  {args.load}')
    logger.info(f'Source path: {args.source}')
    logger.info(f'Target path: {args.target}')
    logger.info(f'Output path: {args.output}')
    logger.info(f'conversion_file: {args.conversion_file}')
    logger.info(f'device: {args.gpu_id}')

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    builder_dir(args.target, args.output)

    # build inferencer
    inferencer = Inferencer(config=config, args=args)

    # inference
    # read conversion list from txt file
    lines = open(args.conversion_file).read().splitlines()
    spect_vc = dir()

    for i in range(0, len(lines)):
        line = line[i]
        source_speech = os.path.join(args.source, line.split()[0])
        target_speech = os.path.join(args.target, line.split()[1])
        inferencer.inference(source_path=source_speech, target_path=target_speech, out_path=args.output, target_root=args.target , seglen=args.seglen, cnt=i+1)
            
