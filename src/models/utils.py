import sys
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd

# Also import local helpers
target_dir = os.path.abspath('..') 

if target_dir not in sys.path:
    sys.path.append(target_dir)
    
from football_lstm import FootballLSTM

# hypertuning of LSTM
def hyperparam_tuning(params: dict, stats_df: pd.DataFrame, train_dataloader: torch.utils.data.DataLoader, 
                          test_dataloader: torch.utils.data.DataLoader):
        
    best_combo = None
    previous_best = None

    for lr in params['learning_rates']:
        for epoch in params['epochs']:
            for layer in params['layers']:
                for h_size in params['h_sizes']:
                    for dropout in params['dropouts']:

                        # non-zero dropout expects num_layers greater than 1 so train with dropout set to 0
                        if layer == 1:
                            model = FootballLSTM(n_features=len(stats_df.columns), hidden_size=h_size, num_layers=layer, dropout=0)
                        else:
                            model = FootballLSTM(n_features=len(stats_df.columns), hidden_size=h_size, num_layers=layer, dropout=dropout)
                                
                        loss_fn = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                            
                        # Train model
                        train_losses, test_losses = model.train_model(
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            n_epochs=epoch
                        )

                        # model evaluation
                        rmse, mae = model.eval_model(test_dataloader)

                        # evaluating if better setup, if so save
                        if previous_best is None or rmse < previous_best:
                            # manually set dropout to 0 if layer is 1
                            best_combo = [lr, epoch, layer, h_size, 0 if layer == 1 else dropout]
                            previous_best = rmse
                            best_train_losses, best_test_losses = train_losses, test_losses

    # save to dictionary for more clear access
    param_dict = {}
    param_dict['learning_rate'] = best_combo[0]
    param_dict['epoch'] = best_combo[1]
    param_dict['layers'] = best_combo[2]
    param_dict['hidden_size'] = best_combo[3]
    param_dict['dropout'] = best_combo[4]

    plt.plot(best_train_losses, color="red", label="Train loss")
    plt.plot(best_test_losses, color="blue", label="Test loss")
    plt.legend()
    plt.title(f"""Hyperparameter Tuning - Train vs Test Loss (lr: {param_dict['learning_rate']}, epoch: {param_dict['epoch']}, 
            layers: {param_dict['layers']}, hidden size: {param_dict['hidden_size']}, dropout: {param_dict['dropout']})""")
    plt.savefig(f"tuning_graphs/"f"lr{param_dict['learning_rate']}_ep{param_dict['epoch']}_ly{param_dict['layers']}_hs{param_dict['hidden_size']}_do{param_dict['dropout']}.png")
    plt.close()
        
    return param_dict