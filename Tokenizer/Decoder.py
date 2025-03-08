from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer


def our_decoder(token_path,
                audio_outpath,
                config_path="./configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
                model_path="./models/wavtokenizer_medium_music_audio_320_24k.ckpt"):

    device = torch.device('cpu')

    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)

    token = torch.load(token_path)
    features = wavtokenizer.codes_to_features(token)
    bandwidth_id = torch.tensor([0])
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
    torchaudio.save(audio_outpath + ".wav", audio_out, sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
