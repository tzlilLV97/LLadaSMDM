from Tokenizer.encoder.utils import convert_audio
import torchaudio
import torch
from Tokenizer.decoder.pretrained import WavTokenizer


def get_tokenizer(config_path="./Tokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
                model_path="./models/wavtokenizer_medium_music_audio_320_24k.ckpt"):
    device = torch.device('cpu')
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)
    return wavtokenizer

def encode_audio(wavtokenizer, audio_path, bandwidth_id=0, device=torch.device('cpu')):
    """ Encode audio file to tokens, return features and discrete code """
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, 24000, 1)
    wav = wav.to(device)
    features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    return features, discrete_code

def encode_and_save_audio(wavtokenizer, audio_path, save_path, bandwidth_id=0, device=torch.device('cpu')):
    """ Encode audio file to tokens and save it """
    features, discrete_code = encode_audio(wavtokenizer, audio_path, bandwidth_id=bandwidth_id, device=device)
    torch.save(discrete_code, save_path)


def decode_audio(wavtokenizer, token_path, audio_outpath, bandwidth_id=0, device=torch.device('cpu'), save=False):
    """ Decode tokens to audio file """
    token = torch.load(token_path)
    features = wavtokenizer.codes_to_features(token)
    bandwidth_id = torch.tensor([bandwidth_id])
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
    if save:
        torchaudio.save(audio_outpath + ".wav", audio_out, sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
    return audio_out