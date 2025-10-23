import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from torch.utils.data import ConcatDataset, DataLoader
import torch.utils.data.sampler as torch_sampler
from collections import defaultdict
from tqdm import tqdm, trange
from exp.learnable_wavelet_domain_sparse_PT import *
import eval_metrics as em

import config_df24
torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)

def initParams():
    parser = config_df24.initParams()
    # Training hyperparameters

    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=16, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    # parser.add_argument('--lr', type=float, default=0.000001, help="learning rate")#FT-10-6
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    
    parser.add_argument('--train_task', type=str, default="speech", choices=["speech"],)
    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="use which loss for basic training")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")

    # generalized strategy 
    parser.add_argument('--SAM', type= bool, default= False, help="use SAM")
    parser.add_argument('--ASAM', type= bool, default= False, help="use ASAM")
    parser.add_argument('--CSAM', type= bool, default= False, help="use CSAM")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds 
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))



        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            json.dump(vars(args), file, indent=4)  

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def shuffle(feat,  labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    # this_len = this_len[shuffle_index]
    return feat, labels



def compute_accuracy(outputs, labels):
    with torch.no_grad():
        if outputs.shape[1] == 1: 
            preds = (torch.sigmoid(outputs) > 0.5).float() 
        else:  
            preds = torch.argmax(outputs, dim=1)  
        
        correct = (preds == labels).float().sum()
        accuracy = correct / labels.shape[0]
    return accuracy


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

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


    
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    
    if args.SAM or args.CSAM:
        feat_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            feat_optimizer,
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    df24_trainset = DF24Dataset(args.df24_train_audio,
                                     args.df24_train_label,
                                     split="Train",
                                     rawboost=False,
                                     rawboost_log=args.rawboost_log,
                                     musanrir=False)
    
    df24_devset = DF24Dataset(args.df24_dev_audio,
                                   args.df24_dev_label,
                                   split="Dev",
                                   rawboost=False,
                                   rawboost_log=args.rawboost_log,
                                   musanrir=False)


    
    if args.train_task == "speech":
        train_set = [df24_trainset]
        dev_set = [df24_devset]
    

    for dataset in train_set:
        print(len(dataset),f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."
    for dataset in dev_set:
        print(len(dataset),f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."

    training_set = ConcatDataset(train_set)
    
    validation_set = ConcatDataset(dev_set)


    trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size),
                            shuffle=False, num_workers=args.num_workers,
                            sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))))                     
    valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size),
                                    shuffle=False, num_workers=args.num_workers,
                                    sampler=torch_sampler.SubsetRandomSampler(range(len(validation_set))))


    trainOri_flow = iter(trainOriDataLoader)
    valOri_flow = iter(valOriDataLoader)

    if args.train_task == "speech":
        weight = torch.FloatTensor([10,1]).to(args.device)   


    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight)

    else:
        criterion = nn.functional.binary_cross_entropy()

    prev_loss = 1e8
    prev_eer = 1
    best_acc = 0.0   # 初始化best_acc为0.0
    monitor_loss = 'base_loss'

        # Initialize tqdm progress bar for epochs
    epoch_bar = tqdm(range(args.num_epochs), desc='Initializing', position=0)
    
    for epoch_num in epoch_bar:
        # Training phase
        feat_model.train()
        trainlossDict = defaultdict(list)
        trainaccDict = defaultdict(list)
        devlossDict = defaultdict(list)
        devaccDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)
        
        # Initialize training progress bar
        train_bar = trange(len(trainOriDataLoader), desc=f'Train Epoch {epoch_num+1}', leave=False, position=1)
        trainOri_flow = iter(trainOriDataLoader)
        
        for i in train_bar:
            try:
                feat, audio_fn, labels = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                feat, audio_fn, labels = next(trainOri_flow)
            
            labels = labels.to(args.device)

            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.mean().backward()
                feat_optimizer.first_step(zero_grad=True)

                disable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                criterion(feat_outputs, labels).mean().backward()
                feat_optimizer.second_step(zero_grad=True)
            else:
                feat_optimizer.zero_grad()
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.backward()
                feat_optimizer.step()

            train_acc = compute_accuracy(feat_outputs, labels)
            
            trainlossDict['base_loss'].append(feat_loss.item())
            trainaccDict['accuracy'].append(train_acc.item())
            
            # Update training progress bar
            train_bar.set_postfix({
                'loss': f"{feat_loss.item():.4f}",
                'avg_loss': f"{np.mean(trainlossDict['base_loss']):.4f}",
                'acc': f"{train_acc.item():.4%}",
                'avg_acc': f"{np.mean(trainaccDict['accuracy']):.4%}"
            })

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(f"{epoch_num}\t{i}\t{trainlossDict[monitor_loss][-1]}\t{trainaccDict['accuracy'][-1]}\n")

        # Validation phase
        feat_model.eval()
        val_bar = trange(len(valOriDataLoader), desc='Validating', leave=False, position=1)
        valOri_flow = iter(valOriDataLoader)
        ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
        
        with torch.no_grad():
            for i in val_bar:
                try:
                    feat, audio_fn, labels = next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    feat, audio_fn, labels = next(valOri_flow)
                
                labels = labels.to(args.device)
                feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs[:, 0]
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]


                val_acc = compute_accuracy(feat_outputs, labels)
                
                ip1_loader.append(feats)
                idx_loader.append(labels)
                devlossDict["base_loss"].append(feat_loss.item())
                devaccDict["accuracy"].append(val_acc.item())
                score_loader.append(score)

                # Update validation progress bar
                val_bar.set_postfix({
                    'loss': f"{feat_loss.item():.4f}",
                    'avg_loss': f"{np.mean(devlossDict['base_loss']):.4f}",
                    'acc': f"{val_acc.item():.2%}",
                    'avg_acc': f"{np.mean(devaccDict['accuracy']):.2%}"
                })

            # Calculate validation metrics
            valLoss = np.nanmean(devlossDict[monitor_loss])
            valAcc = np.nanmean(devaccDict['accuracy'])
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            # Update epoch progress bar
            epoch_bar.set_postfix({
                'trn_loss': f"{np.mean(trainlossDict['base_loss']):.4f}",
                'trn_acc': f"{np.mean(trainaccDict['accuracy']):.4%}",
                'val_loss': f"{valLoss:.4f}",
                'val_acc': f"{valAcc:.4%}",
                'val_eer': f"{val_eer:.4f}",
                'best_acc': f"{best_acc:.4%}"
            })

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(f"{epoch_num}\t{valLoss}\t{val_eer}\t{valAcc}\n")
            print(f"\nVal EER: {val_eer:.4f}, Val Acc: {valAcc:.4%}")

        # Save checkpoints
        # torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'checkpoint', f'anti-spoofing_feat_model_{epoch_num+1}.pt'))
        if (epoch_num + 1) % 10 == 0:
            torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'checkpoint', 
                           f'anti-spoofing_feat_model_{epoch_num+1}.pt'))

        if valLoss < prev_loss:
            torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'best_loss_model.pt'))
            prev_loss = valLoss
            
        if valAcc > best_acc:
            best_acc = valAcc
            torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'best_acc_model.pt'))

        if val_eer < prev_eer:
        # Save the model checkpoint
            torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'best_eer_model.pt'))
            prev_eer = val_eer

    return feat_model


if __name__ == "__main__":
    args = initParams()
    _, _ = train(args)
