import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
from torch.utils.data import DataLoader
from EEGChunkedDatasetBuilder import CustomEEGChunkedDataset
import os

folder_path = "./preprocessed_chunks"
dataset = CustomEEGChunkedDataset(folder_path=folder_path, chunk_size=128)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)



class VectorQuantizer(nn.Module):
    """
    Implements the vector quantization layer.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and calculate loss
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EEGFormer(nn.Module):
    """
    Implements the EEGFormer model based on the paper's description.
    """
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings, commitment_cost):
        super(EEGFormer, self).__init__()
        self.positional_encoding = PositionalEncoding(model_dim)
        encoder_layers = TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.vector_quantizer = VectorQuantizer(num_embeddings, model_dim, commitment_cost)
        decoder_layers = TransformerDecoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, src):
        
        src = self.positional_encoding(src)
        encoded_src = self.transformer_encoder(src)
        quantized, vq_loss, _ = self.vector_quantizer(encoded_src)
        decoded = self.transformer_decoder(quantized, quantized)
        reconstructed = self.fc_out(decoded)
        return reconstructed, vq_loss

# Parameters based on the paper's details

input_dim = 128  # Example input dimension, adjust based on actual preprocessed data dimension
model_dim = 128  # Dimension of model embeddings
num_heads = 8    # Number of heads in multi-head attention mechanism
num_encoder_layers = 6  # Number of Transformer encoder layers
num_decoder_layers = 3  # Number of Transformer decoder layers
num_embeddings = 1024  # Number of embeddings in vector quantization
commitment_cost = 0.25  # Commitment cost used in vector quantization
epochs = 1  # Number of training epochs
log_interval = 10  # Interval for logging training progress

class EEGFormer(nn.Module):
    """
    Implements the EEGFormer model incorporating preprocessing, encoding, vector quantization, and decoding.
    """
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings, commitment_cost):
        super(EEGFormer, self).__init__()
        self.positional_encoding = PositionalEncoding(model_dim)
        encoder_layer = TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.vector_quantizer = VectorQuantizer(num_embeddings, model_dim, commitment_cost)
        decoder_layer = TransformerDecoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, src):
        src = self.positional_encoding(src)
        encoded_src = self.transformer_encoder(src)
        quantized, vq_loss, _ = self.vector_quantizer(encoded_src)
        # Assuming target is same as source for reconstruction in an auto-encoding setup
        decoded = self.transformer_decoder(quantized, quantized)
        reconstructed = self.fc_out(decoded)
        return reconstructed, vq_loss

# Initialize the EEGFormer model with the specified parameters
model = EEGFormer(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, 
                  num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
                  num_embeddings=num_embeddings, commitment_cost=commitment_cost)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    model.train()
    for batch_idx, data_chunk in enumerate(data_loader):
        optimizer.zero_grad()
        reconstructed, vq_loss = model(data_chunk)
        reconstruction_loss = F.mse_loss(reconstructed, data_chunk)
        loss = reconstruction_loss + vq_loss
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data_chunk)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        
    # Save checkpoints to a relative path
    checkpoint_path = f"./checkpoints/eegformer_model_epoch_{epoch}.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # Ensure the directory exists
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
