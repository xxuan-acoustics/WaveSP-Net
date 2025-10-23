import argparse

def initParams():
    parser = argparse.ArgumentParser(description="Configuration for the project")

    parser.add_argument('--seed', type=int, help="Random number seed for reproducibility", default=688)

    # Train & Dev Data folder prepare 



    parser.add_argument("--spoofceleb_train_audio", type=str, help="Path to the training audio for spoofceleb dataset",
                        default='/xxuan/dataset/spoofceleb/data/spoofceleb/flac/train')
    parser.add_argument("--spoofceleb_train_label", type=str, help="Path to the training label for spoofceleb dataset",
                        default="/xxuan/dataset/spoofceleb/data/spoofceleb/metadata/train.csv")  
    parser.add_argument("--spoofceleb_dev_audio", type=str, help="Path to the development audio for spoofceleb dataset",
                        default='/xxuan/dataset/spoofceleb/data/spoofceleb/flac/development')
    parser.add_argument("--spoofceleb_dev_label", type=str, help="Path to the development label for spoofceleb dataset",
                        default="/xxuan/dataset/spoofceleb/data/spoofceleb/metadata/development.csv")  
    parser.add_argument("--spoofceleb_eval_audio", type=str, help="Path to the evaluation audio for spoofceleb dataset",
                        default='/xxuan/dataset/spoofceleb/data/spoofceleb/flac/evaluation')   
    parser.add_argument("--spoofceleb_eval_label", type=str, help="Path to the evaluation label for spoofceleb dataset",
                        default="/xxuan/dataset/spoofceleb/data/spoofceleb/metadata/evaluation.csv") 



    # SSL folder prepare
    parser.add_argument("--xlsr", default="/home/xxuan/speech-deepfake/pre-model/huggingface/wav2vec2-xls-r-300m/")

    
    parser.add_argument("--gpu", type=str, help="GPU index", default="2")
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='/xxuan/results/spoofceleb/Exp19')



    parser.add_argument("-rawboost_log", "--rawboost_log", type=str, help="rawboost_log", required=False, default='5')

    

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