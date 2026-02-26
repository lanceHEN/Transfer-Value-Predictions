# Project Overview

A pipeline that predicts soccer player transfer market values based on historic performance and values.

1. Runs a time series model (LSTM) on player performance to forecast future performance

2. Use predicted performance values as inputs for a regression model (Bayesian) to predict transfer value

Current Idea: To use different models for different positions.

# Repository Overview

## Data Basics Notebook

Provides a overview of the Transfer Market API and Understat API that we use to collect data.

### Transfer Market API

Will be used in future version to collect data on transfer values of players in various leagues to predict their future market values.

### Understat API

Provides us data on player performances on a game level which we use to predict player performances. We will associate player performance with player transfer market value, so that we can use player performance predictions to then also predict player transfer market value.

## Fotball LSTM

A torch class that intializes our LSTM model and trains it with evaluation output in terms of loss during training and testing. Also contains a prediction function that allows us to predict the next k game performances for given player.

## Player Stats

A data support file whose main point is to load the correct data from the Understat API and create datasets for our LSTM. Core function is `get_position_players_stats_df` that creates a dataframe given a desired position (Defender, Midfielder, or Forward) which is then `CustomFootballDataset` uses the dataframe created for given position to create tensors that are passed to LSTM.

## LSTM Notebook

A notebook that loads data, creates custom dataset and loads data for LSTM, splits into train and test (80/20 split) and then trains and evaluates the model.