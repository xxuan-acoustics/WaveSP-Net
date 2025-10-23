from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm, trange
import eval_dataset
import numpy as np
import config
import json
import argparse
from dataset import *

def init():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, default="/xxuan/results/DF24/Exp54")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument("--task", type=str, help="Task type", default="speech", 
                        choices=['speech'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('-m', '--model', help='Model arch', default='WaveSP-Net',
                        choices=['aasist', 'specresnet', 'FT-XLSR-BiMamba', 
                                 'PT-XLSR-BiMamba', 'WPT-XLSR-BiMamba', 
                                 'FourierPT-XLSR-BiMamba',
                                 'WaveSP-Net'])

    parser.add_argument("--df24_eval_audio", type=str, help="Path to the evaluation audio for df24 dataset",
                        default='/home/xxuan/speech-deepfake/All-Type-ADD/datasets/DF24Datasets/test_audio')   
    parser.add_argument("--df24_eval_label", type=str, help="Path to the evaluation label for df24 dataset",
                        default="/home/xxuan/speech-deepfake/All-Type-ADD/protocals/df24_protocols/df24_test_protocol.csv") 
    
    temp_args, _ = parser.parse_known_args()
    
    json_path = os.path.join(temp_args.model_path, 'args.json')
    with open(json_path, 'r') as f:
        json_args = json.load(f)
    
    for key, value in json_args.items():
            if key not in vars(temp_args):
                if isinstance(value, bool):
                    parser.add_argument(f'--{key}', 
                        action='store_true' if value else 'store_false',
                        default=value)
                else:
                    parser.add_argument(f'--{key}', 
                        type=type(value), 
                        default=value)   
    args = parser.parse_args()
    

    if args.batch_size is None:
        args.batch_size = json_args.get('batch_size', None)
    print(args.gpu)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def DF24_test_on_speech(model, args, model_name=""):
    result_dir = os.path.join(args.model_path, 'result')
    os.makedirs(result_dir, exist_ok=True)

    suffix = model_name.replace("_model", "")  
    file_name = f'DF24_{suffix}_speech.txt'
    file_path = os.path.join(result_dir, file_name)
    

    df24_testset = DF24Dataset(args.df24_eval_audio,
                                   args.df24_eval_label,
                                   split="Test",
                                   rawboost=False,
                                   rawboost_log=args.rawboost_log,
                                   musanrir=False)
                                 
    testDataLoader = DataLoader(df24_testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    with torch.no_grad():
        with open(file_path, 'w') as cm_score_file:
            for idx, data_slice in enumerate(tqdm(testDataLoader)):
                waveform, filename,labels = data_slice[0],data_slice[1],data_slice[2] 

                feats, w2v2_outputs = model(waveform)

                scores = F.softmax(w2v2_outputs, dim=1)[:, 0].detach().cpu().numpy()
                for fn, score, label in zip(filename, scores, labels):
                    audio_fn = fn.strip().split('.')[0]
                    label_str = "fake" if label == 1 else "real"
                    cm_score_file.write(f'{audio_fn} {score} {label_str}\n')
                

                       
                
if __name__ == "__main__":
    args = init()
    model_dir = args.model_path
    model_files = {
        # 'best_acc': 'best_acc_model.pt',
        'best_eer': 'best_eer_model.pt'
        # 'best_loss': 'best_loss_model.pt'
    }


    print(f"Initializing model: {args.model}")
    # initialize model
    if args.model == 'aasist':
        feat_model = Rawaasist().cuda()
    if args.model == 'specresnet':
        feat_model = ResNet18ForAudio().cuda()  

    if args.model == "PT-XLSR-BiMamba":  
        feat_model = PT_XLSR_BiMamba(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens,
                                   dropout= args.pt_dropout).cuda()

    if args.model == "WPT-XLSR-BiMamba":  
        feat_model = WPT_XLSR_BiMamba(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens, 
                                   dropout= args.pt_dropout).cuda()

    if args.model == "FourierPT-XLSR-BiMamba":  
        feat_model = FourierPT_XLSR_BiMamba(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens, num_fourier_tokens=args.num_fourier_tokens, 
                                   dropout= args.pt_dropout).cuda()

    if args.model == "WaveSP-Net": 
        feat_model = WaveSP_Net(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens, 
                                   dropout= args.pt_dropout).cuda()
        
        
   
    DF24_test_on_speech(feat_model,args)
