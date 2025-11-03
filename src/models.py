from torch import nn

class Returns1DCNN(nn.Module):
    """1D CNN for predicting next-day returns from historical return sequences."""
    
    def __init__(self, 
                 sequence_length: int = 252,
                 conv_channels: list = [32, 64, 128],
                 kernel_sizes: list = [5, 5, 5],
                 dropout: float = 0.3):
        super(Returns1DCNN, self).__init__()
        
        # Build convolutional layers
        layers = []
        in_channels = 1  # Single feature: returns
        
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size after convolutions
        # After 3 maxpool layers with kernel=2: sequence_length / (2^3)
        reduced_length = sequence_length // (2 ** len(conv_channels))
        fc_input_size = conv_channels[-1] * reduced_length
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Single output: next day return
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        # Add channel dimension: (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x.squeeze(-1)  # (batch_size,)