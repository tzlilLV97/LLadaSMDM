import os
from Encoder import our_encoder
from omegaconf import OmegaConf

# Load config.yaml using OmegaConf
config = OmegaConf.load("config.yaml")

# Define dataset paths
train_path = os.path.join(config.data.train.root, "train_data")
test_path = os.path.join(config.data.test.root, "test_data")
save_path = os.path.join(config.data.train.root, "tokens")

# Ensure the save directory exists
os.makedirs(save_path, exist_ok=True)



def print_token(token):
    data = torch.load(token)
    print(data)
    # Extract tokens from the dictionary
    #tokens = data["tokens"]  # Your saved tokens
    #metadata = data.get("metadata", None)  # Optional metadata


for folder_path in [train_path, test_path]:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Construct full input file path
            file_path = os.path.join(root, file)

            # Remove the file extension (".wav", ".mp3")
            file_name = os.path.splitext(file)[0]

            # Construct full output file path in save_path
            output_file_path = os.path.join(save_path, file_name + ".pt")  # Change ".bin" to the format you need

            # Check if output file already exists
            if not os.path.exists(output_file_path):
                try:
                    # Pass the file to the encoder function
                    our_encoder(file_path, output_file_path)
                    print(f"Processed: {file_path} -> {output_file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            else:
                print(f"File {output_file_path} already exists, skipping...")