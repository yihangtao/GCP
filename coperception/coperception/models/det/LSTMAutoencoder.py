import torch.nn as nn

class TimeDistributed(nn.Module):
    """TimeDistributed wrapper, similar to Keras' TimeDistributed layer."""
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x shape: [batch_size, time_steps, ...]
        batch_size, time_steps = x.size(0), x.size(1)
        
        # Reshape to (batch_size * time_steps, ...)
        reshaped = x.contiguous().view(batch_size * time_steps, *x.size()[2:])
        
        # Apply module
        output = self.module(reshaped)
        
        # Reshape back
        return output.contiguous().view(batch_size, time_steps, *output.size()[1:])

class RepeatVector(nn.Module):
    """RepeatVector layer, similar to Keras' RepeatVector layer."""
    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.n = n

    def forward(self, x):
        # x shape: [batch_size, features]
        return x.unsqueeze(1).repeat(1, self.n, 1)

class LSTMAE(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, num_layers=2, seq_length=5):
        super(LSTMAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Dense layer to process encoded state
        self.encoder_dense = nn.Linear(hidden_dim, hidden_dim)
        
        # RepeatVector to prepare sequence for decoder
        self.repeat_vector = RepeatVector(seq_length)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # TimeDistributed Dense layer for output
        self.time_distributed = TimeDistributed(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Encode sequence
        encoder_output, (hidden, cell) = self.encoder_lstm(x)
        
        # Get encoded state and apply dense layer
        encoded = self.encoder_dense(hidden[-1])
        
        # Repeat vector to create sequence
        decoder_input = self.repeat_vector(encoded)
        
        # Decode
        decoder_output, _ = self.decoder_lstm(decoder_input)
        
        # Apply time distributed layer
        output = self.time_distributed(decoder_output)
        
        return output
