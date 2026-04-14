# Project Overview

A pipeline that predicts soccer player transfer market values at different points in the future based on historic performance and values. Model is specific to (for this project limited to) forward and midfielder positions to get better results for individual positions rather than trying to predict for any player with a single model. 

1. Runs a time series model (LSTM) on player performance to forecast future performance using a player's last k game blocks (aggregated windows of games) to predict how they do in their next game block.

2. Use predicted performance values as inputs for a regression model (Bayesian) to predict future transfer values at various points.

# Repository Overview

## Installation
1. Clone the repository:
```bash
git clone https://github.com/maxizosimov/Transfer-Value-Predictions.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

(Extra) To run streamlit locally:
```bash
ls src
streamlit run app.py
```

## File Structure
```text
src/
|--- data/
    |--- All csvs
|--- models/
    |--- tuning_graphs/
        |--- All graphs from hyperparam tuning
    |---  __init__.py
    |--- football_lstm.py                
    |--- midfielder_bayesian_model.R                    
    |--- forward_bayesian_model.R
    |--- bayesian_utils.R
    |--- lstm.ipynb                      
    |--- utils.py
|--- preprocess/
    |---  __init__.py
    |--- player_stats.py                           
    |--- pull_understat_data.ipynb
    |--- data_basics.ipynb                       
    |--- transfer_performance_match.ipynb
|--- app.py
.gitignore
README.md
requirements.txt
```

## `data`
Contains all csvs used to train and test the models; pulled from understat api to avoid pulling every time. 

## `models`

### `tuning_graphs`
Contains all the graphs produced in hyperparameter tuning efforts to compare best results across runs. 

### `football_lstm.py`
A torch class that intializes our LSTM model and trains it. Also contains a prediction function that allows us to predict the next k game block performances for a given player.

### `lstm.ipynb`
A notebook that showcases the process of loading player data, making a train/test split,
creating torch datasets and dataloaders, and training and evaluating the model. Includes hyperparameter tuning that saves graph of best performance and eventually determines the parameters of the model that is being trained.

### `bayesian_utils.R`
A utils file containing abstractions for running and evaluating bayesian model on given positions as well as a bunch of helper functions such as loading data, winsorizing, one hot encoding, etc. Also contains steps for sampling such as gibbs and getting posteriors as well as checking how well it mixes. Finally contains plotting functionality for plotting player value over time and their posterior predictive distribution.

### `forward_bayesian_model.R`
Utilizes the utils mentioned above together with a setup prior to run and evaluate the forward position data.

### `midfielder_bayesian_model.R`
Utilizes the utils mentioned above together with a setup prior to run and evaluate the midfielder position data.

### `utils.py`
Utils file to abstract the performing of hyperparameter tuning for LSTM in notebook as well as getting the actual values vs predicted values results upon training and evaluation LSTM.

## `preprocess`

### `data_basics.ipynb`
A notebook to display the data in an EDA kind of way, showing both what requesting information from Transfermarkt api and Understat api looks like.

### `player_stats.py`
A data support file to load the correct data from the Understat API and create datasets for our LSTM. Creates a dataframe given a desired position (Defender, Midfielder, or Forward); creates a torch dataset that uses the dataframe created for some player(s) to create input/output tensors that are passed to the LSTM; and has another function to merge with transfermarkt api data to create a full dataset for the Bayesian model.

### `pull_understat_data.ipynb`
Calls understat api, given certain input such as leagues, positions, seasons, and stats which we then save to a csv. This avoids long api calls on every run.

### `transfer_performance_match.ipynb`
Pulls the transfer value data and merges it, adds age and year column, and cleans it to save as csv for use in the Bayesian model.

## app.py
A streamlit application showing two pages: 
1. An overview of the project.
2. An interactive visualization used to show how we would use our bayesian part of project (which also builds on LSTM results). Idea is that you can either input custom inputs or use ones from an existing player from the dataset and see the results of our Bayesian model to be able to assess the predicted transfer value. If you choose a real player, you can tweak the stats using sliders to see, for example, if their expected goals were to increase, how that changes the predictions. 

## APIs

### Transfermarkt API

Collect data on transfer values of players in various leagues. These are used as labels to predict transfer values.

### Understat API

Provides data on player performances on a game level, which we use to predict future performances. We associate player performance, among other features with player transfer market values, so that we can use player performance predictions to predict player transfer market values.