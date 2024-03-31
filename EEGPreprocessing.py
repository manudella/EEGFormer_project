import os
import mne  # Importing the MNE library for EEG data processing
import numpy as np

def preprocess_eeg(eeg_data, sfreq, patch_length, stride):
    """
    Preprocess EEG data by applying FFT, normalizing, and extracting sliding window patches.
    
    Args:
        eeg_data: Numpy array of EEG data.
        sfreq: Sampling frequency of the EEG data.
        patch_length: The length of each patch after splitting.
        stride: The stride with which to apply the sliding window for patch extraction.
        
    Returns:
        A numpy array of preprocessed data.
    """
    # Apply FFT
    fft_data = np.abs(np.fft.fft(eeg_data, axis=1))
    # Normalize the FFT data by channel
    normalized_data = (fft_data - np.mean(fft_data, axis=1, keepdims=True)) / np.std(fft_data, axis=1, keepdims=True)
    
    patches = []
    # Extract sliding window patches for each channel
    for channel in normalized_data:
        all_patches = np.lib.stride_tricks.sliding_window_view(channel, window_shape=(patch_length,))
        stride_indices = np.arange(0, all_patches.shape[0], stride)
        patches.append(all_patches[stride_indices])
    preprocessed_data = np.concatenate(patches, axis=0)
    return preprocessed_data

def save_chunks(preprocessed_data, chunk_size, save_dir, file_prefix):
    """
    Saves the preprocessed data in chunks to the specified directory.
    
    Args:
        preprocessed_data: The data to be saved.
        chunk_size: The size of each chunk.
        save_dir: Directory where chunks will be saved.
        file_prefix: Prefix for the chunk filenames.
    """
    num_chunks = int(np.ceil(preprocessed_data.shape[0] / chunk_size))
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = preprocessed_data[start_idx:end_idx]
        
        # Pad the last chunk if it's smaller than the chunk size
        if chunk.shape[0] < chunk_size:
            chunk = np.pad(chunk, ((0, chunk_size - chunk.shape[0]), (0, 0)), 'constant')
        
        # Save each chunk to a file
        chunk_filepath = os.path.join(save_dir, f"{file_prefix}_chunk_{i}.npy")
        np.save(chunk_filepath, chunk)

def preprocess_and_save_data(folder_path, patch_length=128, stride=64, chunk_size=1024, save_dir='./preprocessed_chunks'):
    """
    Main function to preprocess and save EEG data from EDF files found in a specified folder.
    
    Args:
        folder_path: Path to the folder containing EDF files.
        patch_length: Length of each patch to extract from the EEG data.
        stride: Stride for the sliding window for patch extraction.
        chunk_size: Size of each chunk to save.
        save_dir: Directory to save the preprocessed chunks.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Iterate through each subject folder
    for foldername in os.listdir(folder_path):
        if foldername.startswith("sub"):
            # Process both session 0 and session 1 if present
            for session in ["ses-0", "ses-1"]:
                subject_dir = os.path.join(folder_path, foldername, session, "ieeg")
                if os.path.isdir(subject_dir):
                    for filename in os.listdir(subject_dir):
                        if filename.endswith(".edf"):
                            filepath = os.path.join(subject_dir, filename)
                            raw = mne.io.read_raw_edf(filepath, preload=True)  # Load the EDF file
                            eeg_data = raw.get_data()  # Extract EEG data from the file
                            # Preprocess the data
                            preprocessed = preprocess_eeg(eeg_data, raw.info['sfreq'], patch_length, stride)
                            # Save the preprocessed data in chunks
                            save_chunks(preprocessed, chunk_size, save_dir, filename.split('.')[0])

# Example usage
folder_path = "./data"  # Relative path to the data directory
preprocess_and_save_data(folder_path)

