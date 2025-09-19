import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_channels, img_height, num_classes):
        super(CRNN, self).__init__()
        
        # CNN Part (Feature Extractor)
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        
        # Calculate feature map size
        # This depends on img_height, here we assume 32
        # (img_height / 16 - 1) * 512 = (32/16-1)*512 = 512
        rnn_input_size = 512

        # RNN Part (Sequence Encoder)
        self.rnn = nn.Sequential(
            nn.LSTM(rnn_input_size, 256, bidirectional=True, num_layers=2, batch_first=True),
        )

        # Transcription Head
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        # Pass through CNN
        features = self.cnn(x) # [batch, channels, height, width]
        
        # Reshape for RNN (Map-to-Sequence)
        features = features.squeeze(2) # [batch, channels, width]
        features = features.permute(0, 2, 1) # [batch, width, channels] -> [batch, seq_len, input_size]
        
        # Pass through RNN
        rnn_output, _ = self.rnn(features) # [batch, seq_len, hidden_size*2]
        
        # Pass through Transcription Head
        output = self.fc(rnn_output) # [batch, seq_len, num_classes]
        
        # For CTC Loss, we need [seq_len, batch, num_classes]
        output = output.permute(1, 0, 2)
        
        return output
