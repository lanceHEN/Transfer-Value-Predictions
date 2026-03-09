# Project Overview

A pipeline that predicts soccer player transfer market values at different points in the future based on historic performance and values.

1. Runs a time series model (LSTM) on player performance to forecast future performance (i.e. use a player's last k games to predict how they do in their next game).

2. Use predicted performance values as inputs for a regression model (Bayesian) to predict future transfer values at various points.

Current Idea: To use different models for different positions.

# Repository Overview

## File Structure
```text
src/
├── models/
│   ├── football_lstm.py 
│   ├── lstm.ipynb 
│   └── __init__.py
└── preprocess/
    ├── player_stats.py
    ├── data_basics.ipynb
    └── __init__.py
.gitignore
README.md
requirements.txt
```

### `football_lstm.py`
A torch class that intializes our LSTM model and trains it. Also contains a prediction function that allows us to predict the next k game performances for a given player.

### `lstm.ipynb`
A notebook that showcases the process of loading player data, making a train/test split,
creating torch datasets and dataloaders, and training and evaluating the model.

### `player_stats.py`
A data support file whose main point is to load the correct data from the Understat API and create datasets for our LSTM. Core function is `get_position_players_stats_df` that creates a dataframe given a desired position (Defender, Midfielder, or Forward).`CustomFootballDataset` is a torch dataset that uses the dataframe created for some player(s) to create input/output tensors that are passed to LSTM.

### `data_basics.ipynb`
Provides a overview of the Transfer Market API and Understat API that we use to collect data.

## APIs

### Transfer Market API

Will be used in the future to collect data on transfer values of players in various leagues. These will
be used as labels to predict transfer values.

### Understat API

Provides data on player performances on a game level, which we use to predict future performances. We will associate player performance with player transfer market value, so that we can use player performance predictions to predict player transfer market values.