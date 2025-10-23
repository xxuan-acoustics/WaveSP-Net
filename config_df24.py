import argparse

def initParams():
    parser = argparse.ArgumentParser(description="Configuration for the project")

    parser.add_argument('--seed', type=int, help="Random number seed for reproducibility", default=688)

    # Train & Dev Data folder prepare 

    parser.add_argument("--df24_train_audio", type=str, help="Path to the training audio for df24 dataset",
                        default='/home/xxuan/speech-deepfake/datasets/DF24Datasets/train_audio')
    
    parser.add_argument("--df24_train_label", type=str, help="Path to the training label for df24 dataset",
                        default="/home/xxuan/speech-deepfake/protocals/df24_protocols/df24_train_protocol.csv")  
    
    parser.add_argument("--df24_dev_audio", type=str, help="Path to the development audio for df24 dataset",
                        default='/home/xxuan/speech-deepfake/datasets/DF24Datasets/val_audio')
    parser.add_argument("--df24_dev_label", type=str, help="Path to the development label for df24 dataset",
                        default="/home/xxuan/speech-deepfake/protocals/df24_protocols/df24_val_protocol.csv")  
    parser.add_argument("--df24_eval_audio", type=str, help="Path to the evaluation audio for df24 dataset",
                        default='/home/xxuan/speech-deepfake/datasets/DF24Datasets/test_audio')   
    parser.add_argument("--df24_eval_label", type=str, help="Path to the evaluation label for df24 dataset",
                        default="/home/xxuan/speech-deepfake/protocals/df24_protocols/df24_eval_protocol.csv") 



    # SSL folder prepare
    parser.add_argument("--xlsr", default="/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/pre-model/huggingface/wav2vec2-xls-r-300m/")
    
    parser.add_argument("--gpu", type=str, help="GPU index", default="3")
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='/nlp/nlp-xxuan/icassp_results/DF24/Exp54')


    # countermeasure
    parser.add_argument("--audio_len", type=int, help="raw waveform length", default=64600)
    parser.add_argument('-m', '--model', help='Model arch', default='WaveSP-Net',
                        choices=['aasist', 'specresnet', 'FT-XLSR-BiMamba', 
                                 'PT-XLSR-BiMamba', 'WPT-XLSR-BiMamba', 
                                 'FourierPT-XLSR-BiMamba',
                                 'WaveSP-Net'])
    
    # PT
    parser.add_argument("--prompt_dim", type=int, help="prompt dim", default=1024)
    # parser.add_argument("--num_prompt_tokens", type=int, help="audio dim", default=10)
    parser.add_argument("--pt_dropout", type=float, help="dropout", default=0.1)
    
    # WPT/WSPT/Partial-WSPT
    parser.add_argument("--num_prompt_tokens", type=int, help="audio dim", default=6)
    parser.add_argument("--num_wavelet_tokens", type=int, help="wavelet token", default=4)


    # Fourier-PT
    parser.add_argument("--num_prompt_tokens", type=int, help="audio dim", default=6)
    parser.add_argument("--num_fourier_tokens", type=int, help="fourier token", default=4)
    
    return parser