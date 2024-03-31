import torch
from torch.utils.data import IterableDataset
import os
import numpy as np
import math

class CustomEEGChunkedDataset(IterableDataset):
    def __init__(self, folder_path, chunk_size=128):
        """
        Custom dataset for loading EEG data in chunks.

        Args:
            folder_path (str): Path to the folder containing preprocessed EEG files or chunks.
            chunk_size (int): Size of the chunks to be yielded by the dataset.
        """
        super(CustomEEGChunkedDataset, self).__init__()
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        self.data_files = self._get_data_files()
        self.num_chunks = self._calculate_total_chunks()

    def _calculate_total_chunks(self):
        # Calculate the total number of chunks in the dataset
        total_chunks = 0
        for filepath in self.data_files:
            data = np.load(filepath, mmap_mode='r')
            total_chunks += math.ceil(data.shape[1] / self.chunk_size)
        return total_chunks

    def __len__(self):
        return self.num_chunks

    def _get_data_files(self):
        """
        Gets a list of all relevant files in the specified folder path.

        Returns:
            List of paths to files.
        """
        data_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.npy')]
        return data_files

    def _read_data_chunk(self, filepath):
        """
        Generator to yield chunks of data from a file.

        Args:
            filepath (str): Path to the file to read.
        """
        # Assuming the data is stored in .npy format
        data = np.load(filepath)
        for start_idx in range(0, data.shape[1], self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, data.shape[1])
            yield data[:, start_idx:end_idx]

    def __iter__(self):
        """
        Return the iterator for the dataset.
        """
        file_iters = [self._read_data_chunk(filepath) for filepath in self.data_files]
        for file_iter in file_iters:
            for data_chunk in file_iter:
                yield torch.tensor(data_chunk, dtype=torch.float32)

