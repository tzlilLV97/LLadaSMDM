from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer
from Decoder import our_decoder
from Encoder import our_encoder

# Encode audio file to tokens and Decode it back

audio_path = r"C:\Projects\WavTokenizer\TrainTokens\AudioData\1073150.low.mp3"
save_path = r"C:\Projects\WavTokenizer\TrainTokens\tokenTest"


# Load config.yaml using OmegaConf for structured access
config = OmegaConf.load("config.yaml")


#Train and test dataset paths
train_path = config.data.train.root + r"\train_data"
test_path = config.data.test.root + r"\test_data"


save_path = config.data.train.root + r"\tokens"

#Mkdir if not exists for the tokens
os.makedirs(save_path, exist_ok=True)
our_encoder(audio_path, save_path)

token_path = save_path
audio_outpath = "./after_tokenizing"

our_decoder(token_path, audio_outpath)




