import musicnet

# Load config.yaml using OmegaConf for structured access
config = OmegaConf.load("config.yaml")



def download_tokenizer():
    """download the tokenizer model"""
    import requests
    import os
    from tqdm import tqdm

    # Define URL and destination folder
    url = "https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/resolve/main/wavtokenizer_medium_music_audio_320_24k.ckpt"
    save_dir = "models"
    filename = os.path.join(save_dir, "wavtokenizer_medium_music_audio_320_24k.ckpt")

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Stream the download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

    print(f"\nModel downloaded successfully and saved at: {filename}")




def Download_dataset():
    """Initialize the MusicNet dataset in order to download it"""
    musicnet.MusicNet(
        root=config.data.train.root,
        train=config.data.train.train,
        download=True,
        window=config.data.train.window,
    )
