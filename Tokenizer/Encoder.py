from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer


def our_encoder(audio_path,
                save_path,
                config_path="./Tokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
                model_path="./models/wavtokenizer_medium_music_audio_320_24k.ckpt"):

    device = torch.device('cpu')

    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    wavtokenizer = wavtokenizer.to(device)

    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, 16000, 1)
    bandwidth_id = torch.tensor([0])
    wav = wav.to(device)
    _, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

    token = discrete_code[:, :, :1024]
    print(token.shape)
    torch.save(token, save_path)

