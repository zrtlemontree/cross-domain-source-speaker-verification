import argparse
import yaml
from tools import HParams
from againvc.inference_batch import process_data_againvc
from freevc.convert_batch import process_data_freevc
from vqmivc.convert_batch import process_data_vqmivc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='The train config with respect to the model resumed.', default='./config/vqmivc.yaml')
    args = parser.parse_args()
    args = read_params(args.config)
    return args

def read_params(file):
    with open(file, "r", encoding="utf-8") as f:
        result = f.read()
        result = yaml.load(result, Loader=yaml.FullLoader)
    args = HParams(**result)
    return args

if __name__ == "__main__":
    args = get_args()
    VC_type = args.VC_type
    if VC_type == "AgainVC":
        process_data_againvc(args)
    elif VC_type == "FreeVC": 
        process_data_freevc(args)
    elif VC_type == "VQMIVC":
        process_data_vqmivc(args)
