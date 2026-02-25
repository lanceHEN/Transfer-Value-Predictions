import torch
import torch.nn as nn

class FootballLSTM(nn.Module):
    
    
    def __init__(self, n_features: int, hidden_size: int):
        """
        Initializes a FootballLSTM, with the given hidden size, to work with the
        given number of input features. Note that n_features defines both
        the input and output dimensions for the underlying LSTM implementation. 
        """
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        """
        Produces a prediction for the next game after the games represented in x.
        
        x should have shape (batch_size, window_size, n_features).
        
        Output will have shape (batch_size, n_features)
        """
        lstm_out, _ = self.lstm(x) # (batch_size, window_size, hidden_size)
        last_hidden_state = lstm_out[:, -1, :] # (batch_size, hidden_size)
        output = self.fc(last_hidden_state) # (batch_size, n_features)
        return output
    
    def predict_next_k(self, x: torch.Tensor, k: int):
        """
        Produces a prediction for each of the next k games after the games
        represented in x.
        
        x should have shape (batch_size, window_size, n_features).
        
        Output will have shape (batch_size, k, n_features)
        """
        predictions = []

        inpt = x
        for _ in range(10):
            out = self(inpt) # (batch_size, n_features)
            predictions.append(out)
            # Keep all but 1st, and add next
            inpt = torch.cat((inpt[:,1:,:], out.unsqueeze(0)), dim=1) # (batch_size, window_size, n_features)
            
        return torch.stack(predictions).transpose(0,1) # (batch_size, k, n_features)