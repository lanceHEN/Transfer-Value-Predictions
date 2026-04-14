import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict

# have GPU available to speed up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FootballLSTM(nn.Module):
    
    
    def __init__(self, n_features: int, hidden_size: int, num_layers: int=1, dropout: float=0.3):
        """
        Initializes a FootballLSTM, with the given hidden size, to work with the
        given number of input features. Note that n_features defines both
        the input and output dimensions for the underlying LSTM implementation. 
        """
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        """
        Produces a prediction for the next game block after the game blocks represented in x.
        
        x should have shape (batch_size, blocks_per_input, n_features).
        
        Output will have shape (batch_size, n_features)
        """
        lstm_out, _ = self.lstm(x) # (batch_size, blocks_per_input, hidden_size)
        last_hidden_state = lstm_out[:, -1, :] # (batch_size, hidden_size)
        output = self.fc(last_hidden_state) # (batch_size, n_features)
        return output
    
    @torch.no_grad()
    def predict_next_k(self, x, k: int):
        """
        Produces a prediction for each of the next k game blocks after the game blocks
        represented in x.
        
        x should have shape (batch_size, blocks_per_input, n_features).
        
        Output will have shape (batch_size, k, n_features)
        """
        predictions = []

        inpt = x
        for _ in range(k):
            out = self(inpt) # (batch_size, n_features)
            predictions.append(out)
            # Keep all but 1st, and add next
            inpt = torch.cat((inpt[:,1:,:], out.unsqueeze(1)), dim=1) # (batch_size, blocks_per_input, n_features)
            
        return torch.stack(predictions).transpose(0,1) # (batch_size, k, n_features)
    
    def train_model(self, optimizer, loss_fn, train_dataloader, test_dataloader, n_epochs: int=10, test_every: int=1):
        """
        Trains the model for n_epochs using the given optimizer and loss function,
        on the given training dataloader. After each test_every epochs, evaluates on the test
        dataloader as well. Returns the list of train and test losses, averaged
        per-batch.
        """
        train_losses = []
        test_losses = []
        
        for epoch in range(n_epochs):
            self.train()
            train_loss = 0
            for X_batch, y_batch, _ in train_dataloader:
                # to be able to run on GPU
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = loss_fn(outputs, y_batch)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
            train_losses.append(train_loss / len(train_dataloader))
                
            if epoch % test_every == 0:
                test_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch, _ in test_dataloader:
                        # to be able to run on GPU
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        outputs = self(X_batch)
                        
                        loss = loss_fn(outputs, y_batch)
                        test_loss += loss.item()
            
            test_losses.append(test_loss / len(test_dataloader))
            
        return train_losses, test_losses

    @torch.no_grad()
    def get_test_preds(self, test_dataloader):
        """
        For the given test dataframe, produced torch tensors for all true and
        predicted outputs, each of shape (test_size, n_features). Also returns the number
        of steps ahead for each output
        """
        self.eval()
        
        # make a dict from player to x and y - then can do predict next k
        player_data = defaultdict(lambda: {"X": [], "y": []})
    
        for X_batch, y_batch, player_ids in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            for i, id in enumerate(player_ids):
                # Make sure to .item() on id otherwise you get hashing issues
                player_data[id.item()]["X"].append(X_batch[i])
                player_data[id.item()]["y"].append(y_batch[i])
                
        # For each player do predict next k and add to running outputs
        #print(player_data)
        
        y_preds = [] # (n_batches, batch_size, n_features)
        y_trues = [] # (n_batches, batch_size, n_features)
        steps = [] 
        
        for id, data_dict in player_data.items():
            # Start from first however many blocks
            start = data_dict["X"][0].unsqueeze(0)
            n_steps = len(data_dict["y"])
            
            preds = self.predict_next_k(start, n_steps).squeeze(0)
            trues = torch.stack(data_dict["y"])
            
            y_preds.append(preds)
            y_trues.append(trues)
            steps.append(torch.arange(1, n_steps + 1)) # 1 up to n steps
            
        y_preds = torch.cat(y_preds).detach().cpu().numpy() # (test_size, n_features)
        y_trues = torch.cat(y_trues).detach().cpu().numpy() # (test_size, n_features)
        steps = torch.cat(steps).detach().cpu().numpy()
        
        return y_trues, y_preds, steps
    
    @torch.no_grad()
    def look_ahead_errors(self, test_dataloader):
        """
        Produces the RMSE per number of blocks into the future.
        """
        y_trues, y_preds, steps = self.get_test_preds(test_dataloader)
        
        results = {}
        for step in np.unique(steps):
            mask = steps == step
            rmse = mean_absolute_error(y_trues[mask], y_preds[mask])
            results[step] = rmse

        return results
    
    @torch.no_grad()
    def eval_model(self, test_dataloader):
        """
        Evaluates model performance on the given test dataloader by making
        autoregressive (NOT ONE STEP AHEAD) forecasts, returning
        a dict containing the overall RMSE and MAE at key 'Overall', as
        well as the individual columns, 0-indexed.
        """
        y_trues, y_preds, _ = self.get_test_preds(test_dataloader)
 
        results_dict = {}
        
        for i in range(y_preds.shape[1]):
            results_dict[i] = (
                root_mean_squared_error(y_trues[:,i], y_preds[:,i]),
                mean_absolute_error(y_trues[:,i], y_preds[:,i])
            )

        results_dict["Overall"] = (
            root_mean_squared_error(y_trues, y_preds),
            mean_absolute_error(y_trues, y_preds)
        )

        return results_dict
        
    @torch.no_grad()
    def eval_model_on_player(self, player_stats_df: pd.DataFrame, blocks_per_input: int=10, title: str = ""):
        """
        Evaluates model performance on a particular player DataFrame. Uses the
        first blocks_per_input game blocks for fitting and the rest for evaluation, plotting
        each actual vs. predicted stat over time.
        
        Also prints RMSE and MAE for the combined metrics, and for each metric
        by itself.
        """
        self.eval()
        x = torch.tensor(player_stats_df.values[:blocks_per_input], dtype=torch.float32)
        
        # Have to unsqueeze to add batch dimension
        x = x.unsqueeze(0).to(device)
        
        y_trues = torch.tensor(player_stats_df.values[blocks_per_input:], dtype=torch.float32)
        y_trues = y_trues.to(device)
        
        y_preds = self.predict_next_k(x, len(player_stats_df) - blocks_per_input)
        
        # Get rid of batch dimension
        y_preds = y_preds.squeeze(0)
        
        # Convert to numpy to be able to use sklearn
        y_trues_np = y_trues.detach().cpu().numpy()
        y_preds_np = y_preds.detach().cpu().numpy()

        # Basic metrics
        print(f"RMSE: {root_mean_squared_error(y_trues_np, y_preds_np)}")
        print(f"MAE: {mean_absolute_error(y_trues_np, y_preds_np)}")
        
        fig, ax = plt.subplots(player_stats_df.shape[1], 1, layout="constrained", figsize=(20, 5 * len(player_stats_df.columns)))
        
        plt.suptitle(title, fontsize=15, fontweight="bold")
        
        x = range(1, len(y_trues) + 1) # To get plot starting from 1
        
        # Get plots for every stat
        for i, stat in enumerate(player_stats_df.columns):
            
            stat_actuals = y_trues[:,i]
            stat_predictions = y_preds[:,i]
            
            # Per-stat metrics
            print(f"{stat} RMSE: {root_mean_squared_error(stat_actuals, stat_predictions)}")
            print(f"{stat} MAE: {mean_absolute_error(stat_actuals, stat_predictions)}")
            
            ax[i].plot(x, stat_actuals, color="blue", label=f"Actual {stat}")
            ax[i].plot(x, stat_predictions, color="red", label=f"Predicted {stat}")
            ax[i].legend(fontsize=13, loc="lower right")
            ax[i].grid()
            ax[i].set_xlabel("Game Blocks Ahead", fontsize=15)
            ax[i].set_ylabel(f"{stat}", fontsize=15)
            ax[i].tick_params(axis='both', labelsize=12)
        
        plt.show()
   