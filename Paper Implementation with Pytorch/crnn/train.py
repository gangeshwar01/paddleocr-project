import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CRNN
from ctc_codec import CTCCodec
# from dataset import CRNNDataset # You would need to implement this

def train_crnn():
    # --- Config ---
    epochs = 100
    lr = 0.0005
    batch_size = 16
    img_height = 32
    img_channels = 1 # Grayscale
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    num_classes = len(alphabet) + 1 # +1 for blank
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Setup ---
    print(f"Using device: {device}")
    codec = CTCCodec(alphabet)
    model = CRNN(img_channels, img_height, num_classes).to(device)
    criterion = nn.CTCLoss(blank=codec.blank_idx, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train_dataset = CRNNDataset(label_file="path/to/labels.txt")
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("WARNING: Using dummy data. Implement CRNNDataset for real training.")

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(range(100), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            # --- DUMMY DATA ---
            # Replace with: images, texts = batch
            dummy_images = torch.randn(batch_size, img_channels, img_height, 200).to(device)
            dummy_texts = ["hello" for _ in range(batch_size)]
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(dummy_images) # [seq_len, batch, num_classes]
            log_probs = nn.functional.log_softmax(preds, dim=2)
            
            # Prepare for CTC Loss
            texts_encoded = [codec.encode(t) for t in dummy_texts]
            target_lengths = torch.IntTensor([len(t) for t in texts_encoded])
            targets = torch.cat([torch.IntTensor(t) for t in texts_encoded])
            input_lengths = torch.full(size=(batch_size,), fill_value=preds.size(0), dtype=torch.long)
            
            # Calculate loss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / 100
        print(f"Epoch {epoch+1} Summary: Average Loss = {avg_loss:.4f}")
        
    print("Training finished.")
    torch.save(model.state_dict(), "crnn_final.pth")

if __name__ == '__main__':
    train_crnn()
