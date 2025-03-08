import os
from Decoder import our_decoder

# Path to the target folder
folder_path = r"C:\Projects\WavTokenizer\TrainTokens\Tokens"
save_path = r"C:\Projects\WavTokenizer\TrainTokens\DecodedTokens\Decoded-"

# Iterate over all files in the folder (including subfolders)
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Construct the full path to the file
        file_path = os.path.join(root, file)

        # Construct the output file path
        output_file_path = save_path + file[6:] + '.wav'

        # Check if the output file already exists
        if not os.path.exists(output_file_path):
            try:
                # Pass the file name to the function
                our_decoder(file_path, output_file_path)
            except Exception as e:
                # Print the file name and error message
                print(f"Failed to process file: {file_path}")
                print(f"Error: {e}")
        else:
            print(f"File {output_file_path} already exists, skipping...")


