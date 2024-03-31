## Downstream Task
from torch import nn
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
from EEGFormer import EEGFormer
from torch.utils.data import DataLoader
from EEGChunkedLabeledDatasetBuilder import EEGChunkedLabeledDataset

folder_path = "./preprocessed_chunks_labeled"
dataset = EEGChunkedLabeledDataset(folder_path=folder_path, chunk_size=128, labels_ext='.npz')
N_classes = dataset.num_classes
data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

# Parameters based on the paper's details
input_dim = 128  # input dimension, adjust based on actual preprocessed data dimension
model_dim = 128  # Dimension of model embeddings
num_heads = 8    # Number of heads in multi-head attention mechanism
num_encoder_layers = 6  # Number of Transformer encoder layers
num_decoder_layers = 3  # Number of Transformer decoder layers
num_embeddings = 1024  # Number of embeddings in vector quantization
commitment_cost = 0.25  # Commitment cost used in vector quantization
epochs = 1  # Number of training epochs
log_interval = 10  # Interval for logging training progress

class EEGFormerForClassification(EEGFormer):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings, commitment_cost, N_classes):
        super(EEGFormerForClassification, self).__init__(input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings, commitment_cost)
        # Modify the output layer for classification
        self.fc_out = nn.Linear(model_dim, N_classes)  # Change for N_classes
    
    def forward(self, src):
        # Keep the forward pass until the decoder
        src = self.positional_encoding(src)
        encoded_src = self.transformer_encoder(src)
        quantized, vq_loss, _ = self.vector_quantizer(encoded_src)
        decoded = self.transformer_decoder(quantized, quantized)
        # Output layer for classification
        output = self.fc_out(decoded)
        return F.log_softmax(output, dim=-1), vq_loss  # Use log_softmax for NLLLoss

def calculate_accuracy(output, labels):
    _, predicted = torch.max(output.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

# Load the pre-trained model weights
pretrained_model_path = "./checkpoints/eegformer_model_epoch_1.pth"
# Initialize the EEGFormer model for classification with the correct number of output classes
model_for_finetuning = EEGFormerForClassification(input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, num_embeddings, commitment_cost, N_classes)

# Load the pre-trained model weights, excluding 'fc_out' to avoid size mismatch
pretrained_dict = torch.load(pretrained_model_path)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc_out' not in k}  # Exclude 'fc_out' layer
model_for_finetuning.load_state_dict(pretrained_dict, strict=False)  # Load remaining parameters

# Proceed with fine-tuning
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_for_finetuning.parameters()), lr=1e-4)
criterion = nn.NLLLoss()

for epoch in range(1, epochs + 1):
    model_for_finetuning.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    # Setup the progress bar
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch}/{epochs}") as pbar:
        for batch_idx, (data_chunk, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            output, vq_loss = model_for_finetuning(data_chunk)
            output = output.mean(dim=-1) 
            output = F.log_softmax(output, dim=1)
            labels = labels.squeeze(1)  
            classification_loss = criterion(output, labels)
            loss = classification_loss + vq_loss
            loss.backward()
            optimizer.step()
            # Accumulate loss and accuracy
            running_loss += loss.item()
            acc = calculate_accuracy(output, labels)
            correct_predictions += (output.argmax(1) == labels).sum().item()
            total_predictions += labels.size(0)
            pbar.update(1) 
            pbar.set_postfix(Loss=running_loss/(batch_idx+1), Accuracy=acc, LearningRate=optimizer.param_groups[0]['lr'])                        
    
    epoch_loss = running_loss / len(data_loader)
    epoch_acc = 100 * correct_predictions / total_predictions
    print(f"End of Epoch {epoch}: Avg Loss: {epoch_loss:.4f}, Avg Accuracy: {epoch_acc:.2f}%")
    checkpoint_path = f"./checkpoints/eegformer_finetuned_model_epoch_{epoch}.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True) 
    torch.save(model_for_finetuning.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    

