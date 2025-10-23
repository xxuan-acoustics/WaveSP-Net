from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import argparse
from dataset import *
from model import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

torch.multiprocessing.set_start_method('spawn', force=True)


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/xxuan/results/spoofceleb/Exp13")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument("--task", type=str, help="Task type", default="speech", choices=['speech'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--xlsr", default="/home/xxuan/speech-deepfake/pre-model/huggingface/wav2vec2-xls-r-300m/")
    # pt
    parser.add_argument("--prompt_dim", type=int, help="prompt dim", default=1024)
    parser.add_argument("--num_prompt_tokens", type=int, help="audio dim", default=6)
    # parser.add_argument("--num_prompt_tokens", type=int, help="audio dim", default=10)
    parser.add_argument("--pt_dropout", type=float, help="dropout", default=0.1)
    
    # wpt/wspt
    parser.add_argument("--num_wavelet_tokens", type=int, help="wavelet token", default=4)

    # fourier_pt
    parser.add_argument("--num_fourier_tokens", type=int, help="fourier token", default=4)

    parser.add_argument('-m', '--model', help='Model arch', default='WaveSP-Net',
                        choices=['aasist', 'specresnet', 'FT-XLSR-BiMamba', 
                                 'PT-XLSR-BiMamba', 'WPT-XLSR-BiMamba', 
                                 'FourierPT-XLSR-BiMamba',
                                 'WaveSP-Net'])

    
    parser.add_argument("--spoofceleb_eval_audio", type=str, help="Path to the evaluation audio for spoofceleb dataset",default='/xxuan/dataset/spoofceleb/data/spoofceleb/flac/evaluation')   

    parser.add_argument("--spoofceleb_eval_label", type=str, help="Path to the evaluation label for spoofceleb dataset",default="/xxuan/dataset/spoofceleb/data/spoofceleb/metadata/evaluation.csv") 

    args = parser.parse_args()

    print(f"Using GPU: {args.gpu}")


    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def check_model_output_format(model, sample_data):
    with torch.no_grad():
        feats, outputs = model(sample_data)
        probs = F.softmax(outputs, dim=1)
        print(f"Model output shape: {outputs.shape}")
        print(f"Sample softmax outputs: {probs[0]}")
    return outputs, probs


def get_fake_probability(w2v2_outputs):

    probs = F.softmax(w2v2_outputs, dim=1)
    fake_probs = probs[:, 1].detach().cpu().numpy()  
    return fake_probs

def spoofceleb_test_on_speech(model, args, model_name="", threshold=0.5):

    results_dir = os.path.join(args.model_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    suffix = model_name.replace("_model", "")
    file_name = f'spoofceleb_{suffix}_speech.txt' if suffix else 'spoofceleb_speech.txt'
    file_path = os.path.join(results_dir, file_name)

    spoofceleb_testset = spoofceleb(
        args.spoofceleb_eval_audio,
        args.spoofceleb_eval_label,
        # split="Test",
        rawboost=False,
        rawboost_log=getattr(args, 'rawboost_log', False),
        musanrir=False
    )

    test_loader = DataLoader(spoofceleb_testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    y_true_all, y_score_all, filenames_all = [], [], []

    with torch.no_grad():
        with open(file_path, 'w') as cm_score_file:
            for data_slice in tqdm(test_loader):
                waveform, filename, labels = data_slice[0].to(args.device), data_slice[1], data_slice[2]
                
                feats, w2v2_outputs = model(waveform)
                
                probs = F.softmax(w2v2_outputs, dim=1)

                if len(probs[0]) == 2:  
       
                    scores = probs[:, 1].detach().cpu().numpy()  # prob(fake)
                else:
                    raise ValueError(f"Unexpected output shape: {probs.shape}")

                for fn, score, label in zip(filename, scores, labels):
                    audio_fn = fn.strip().split('.')[0]
                    label_str = "fake" if int(label) == 1 else "real"
                    cm_score_file.write(f'{audio_fn} {score} {label_str}\n')

                    y_true_all.append(int(label))   # 1=fake, 0=real
                    y_score_all.append(float(score))
                    filenames_all.append(fn)

    y_pred = (np.array(y_score_all) >= threshold).astype(int)
    

    acc = accuracy_score(y_true_all, y_pred)
    prec = precision_score(y_true_all, y_pred, zero_division=0)
    rec = recall_score(y_true_all, y_pred, zero_division=0)
    f1 = f1_score(y_true_all, y_pred, zero_division=0)
    auc = roc_auc_score(y_true_all, y_score_all)



    acc_path = os.path.join(results_dir, "results.txt")
    with open(acc_path, "w") as f:
        f.write(f"Samples: {len(y_true_all)}\n")
        f.write(f"Threshold: {threshold:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")

    print(f"[spoofceleb] Scores saved to: {file_path}")
    print(f"[spoofceleb] Metrics saved to: {acc_path}")
    
    return {
        'accuracy': acc,
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'y_true': y_true_all,
        'y_scores': y_score_all
    }


if __name__ == "__main__":

    args = init()
    model_dir = args.model_path

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_files = {
        'best_eer': 'best_eer_model.pt'
    }
    
    m = args.model
    if m == 'aasist':
        feat_model = Rawaasist()

    elif m == 'specresnet':
        feat_model = ResNet18ForAudio()

    elif m == 'fr-w2v2aasist':
        feat_model = XLSRAASIST(model_dir=args.xlsr, freeze=True)


    elif m == "PT-XLSR-BiMamba":  
        feat_model = PT_XLSR_BiMamba(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens,
                                   dropout= args.pt_dropout)

    elif m == "WPT-XLSR-BiMamba":  
        feat_model = WPT_XLSR_BiMamba(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens, 
                                   dropout= args.pt_dropout)

    elif m == "FourierPT-XLSR-BiMamba":  
        feat_model = FourierPT_XLSR_BiMamba(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens, num_fourier_tokens=args.num_fourier_tokens, 
                                   dropout= args.pt_dropout)

    elif m == "WaveSP-Net": 
        feat_model = WaveSP_Net(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens, 
                                   dropout= args.pt_dropout)

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    feat_model = feat_model.to(device)

    def _load_state_dict_robust(model, ckpt):

        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            state = ckpt['model']
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            state = ckpt

        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            if k.startswith('module.'):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)

    for key, filename in model_files.items():
        ckpt_path = os.path.join(model_dir, filename)
        # ckpt_path = "/xxuan/results/DF24/Exp23/checkpoint/anti-spoofing_feat_model_50.pt"
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found. Skipping...")
            continue

        print(f"Loading {key} model from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        _load_state_dict_robust(feat_model, checkpoint)

        feat_model.eval()
        spoofceleb_test_on_speech(feat_model, args, model_name=key)

