import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import librosa
from torch.utils.data.dataloader import default_collate
import glob
import random
import numpy
import soundfile
from scipy import signal
from RawBoost import process_Rawboost_feature

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath,sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

def pad_dataset(wav, audio_length=64600):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = audio_length
    if waveform_len >= cut:
        waveform = waveform[:cut]
    else:
        # need to pad
        num_repeats = int(cut / waveform_len) + 1
        waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
    
    return waveform

class AudioAugmentor:
    def __init__(self, rir_path='/RIRS_NOISES', musan_path = '/musan'):
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = self._load_noiselist(musan_path)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*/*.wav'))

    def _load_noiselist(self, musan_path):
        noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            category = file.split('/')[-3]
            if category not in noiselist:
                noiselist[category] = []
            noiselist[category].append(file)
        return noiselist

    def add_rev(self, audio, audio_length):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :audio_length]

    def add_noise(self, audio, noisecat, audio_length):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = audio_length
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
    
    

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

class spoofceleb(Dataset):
    
    def __init__(self, path_to_features, path_to_protocol, 
                 rawboost=True, musanrir=False, audio_length=64600, rawboost_log=5):
        
        super(spoofceleb, self).__init__()

        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.rawboost_log = rawboost_log
        self.rawboost = rawboost
        self.musanrir = musanrir
        self.AudioAugmentor = AudioAugmentor() 

        self.files = []
        with open(path_to_protocol, 'r') as f:

            next(f)  
            
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue  
                
                file_path = parts[0]  
                attack_type = parts[2] 
                
                label = 0 if attack_type == "a00" else 1
                
                self.files.append((file_path, label))

        print(f"Successfully loaded {len(self.files)} SpoofCeleb utterances")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        
        full_path = os.path.join(self.path_to_features, file_path)
        
        waveform, sr = torchaudio.load(full_path)
        waveform = waveform.squeeze(0)  

        if self.rawboost:
            waveform_np = waveform.numpy()
            waveform_np = process_Rawboost_feature(
                waveform_np, 
                sr=sr, 
                rawboost_algo=self.rawboost_log
            )
            waveform = torch.from_numpy(waveform_np)

        waveform = pad_dataset(waveform, self.audio_length)
        
        if self.musanrir:
            waveform = self._apply_augmentation(waveform, self.audio_length)
        
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return waveform, filename, label

    def _apply_augmentation(self, waveform, audio_length):
        augtype = random.randint(0, 4)
        
        if augtype == 0:
            return waveform
        elif augtype == 1:
            return torch.tensor(
                self.AudioAugmentor.add_rev(waveform.unsqueeze(0).numpy(), audio_length)
            ).squeeze(0)
        else:
            noise_type = {2: 'noise', 3: 'speech', 4: 'music'}[augtype]
            return torch.tensor(
                self.AudioAugmentor.add_noise(waveform.unsqueeze(0).numpy(), noise_type, audio_length)
            ).squeeze(0)

    def collate_fn(self, samples):
        waveforms, filenames, labels = zip(*samples)
        waveforms = torch.stack(waveforms)
        labels = torch.tensor(labels)
        return waveforms, filenames, labels




class DF24Dataset(Dataset):
    
    def __init__(self, path_to_features, path_to_protocol, 
                 split="Train", 
                 rawboost=False, rawboost_log=5, 
                 musanrir=False,
                 target_sr=16000,  
                 audio_length=64000):  

        super(DF24Dataset, self).__init__()

        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.rawboost = rawboost
        self.rawboost_log = rawboost_log
        self.musanrir = musanrir
        self.split = split
        self.target_sr = target_sr
        self.audio_length = audio_length
        
        self.use_augmentation = self.rawboost and self.musanrir and (split == "Train")

        if self.use_augmentation:
            self.AudioAugmentor = AudioAugmentor()

        self.files = []
        with open(path_to_protocol, 'r') as f:
            next(f)
            
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 5:
                    continue  
                
                file_name = parts[0] 
                ground_truth = parts[2].strip()  
                finetuning_set = parts[4].strip() 
                
                if finetuning_set != self.split:
                    continue
                
                if ground_truth.lower() in ['real', 'fake']:
                    label = 0 if ground_truth == "Real" else 1
                else:
                    label = -1  

                self.files.append((file_name, label))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name, label = self.files[idx]
        
        full_path = os.path.join(self.path_to_features, file_name)
        
        waveform, sr = torchaudio.load(full_path)
        

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  
        

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
        
        if self.use_augmentation:
            if self.rawboost:
                waveform_np = waveform.numpy()
                waveform_np = process_Rawboost_feature(waveform_np, sr=self.target_sr, rawboost_algo=self.rawboost_log)
                waveform = torch.from_numpy(waveform_np)
            if self.musanrir:
                waveform = self._apply_augmentation(waveform)
            

        current_length = len(waveform)
        if current_length < self.audio_length:
            padding = self.audio_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > self.audio_length:
            start = (current_length - self.audio_length) // 2
            waveform = waveform[start:start+self.audio_length]
        
        filename = os.path.splitext(file_name)[0]
        return waveform, filename, label

    def _apply_augmentation(self, waveform):
        augtype = random.randint(0, 4)
        
        if augtype == 0:
            return waveform
        elif augtype == 1:
            return torch.tensor(
                self.AudioAugmentor.add_rev(waveform.unsqueeze(0).numpy(), waveform.shape[0])
            ).squeeze(0)
        else:
            noise_type = {2: 'noise', 3: 'speech', 4: 'music'}[augtype]
            return torch.tensor(
                self.AudioAugmentor.add_noise(waveform.unsqueeze(0).numpy(), noise_type, waveform.shape[0])
            ).squeeze(0)

    def collate_fn(self, samples):
        waveforms, filenames, labels = zip(*samples)
        
        waveforms = [
            wf if len(wf) == self.audio_length
            else (torch.nn.functional.pad(wf, (0, self.audio_length - len(wf))) if len(wf) < self.audio_length
                  else wf[:self.audio_length])
            for wf in waveforms
        ]
        waveforms = torch.stack(waveforms)
        labels = torch.tensor(labels)
        return waveforms, filenames, labels

 






