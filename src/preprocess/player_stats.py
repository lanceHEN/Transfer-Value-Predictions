from typing import List, Set

import pandas as pd
from understatapi import UnderstatClient
import torch
from torch.utils.data import Dataset

"""
This module contains basic helper functions or classes to load in or store
player stats using understat.
"""

def get_player_ids(understat: UnderstatClient, positions: Set[str], league: str = "EPL", season: str = "2025") -> List[str]:
    """
    Gets a list of all player ids for the given league, season, and positions.
    """
    players = understat.league(league=league).get_player_data(season=season)
    
    # Filter by position
    pos_players = list(filter(lambda p: any(pp in positions for pp in p["position"].split(" ")), players))
    return [p["id"] for p in pos_players]
    
def get_player_stats_df(understat: UnderstatClient, player_id: str,
                      stats: List[str]) -> pd.DataFrame:
    """
    Produces a dataframe with per-90 stats for every game played by the given
    player_id, for any club or season played. Indexes by date.
    """
    # Get all player matches
    player_matches = understat.player(player=player_id).get_match_data()
        
    # Convert to dataframe
    player_matches_df = pd.DataFrame(player_matches)
        
    # Sort by date
    player_matches_df = player_matches_df.sort_values(by="date")
        
    # Per-90 stats over games
    per_90_stats = list(map(lambda s: f"{s}_per_90", stats))

    # Stats/90 for each
    for stat, per_90_stat in zip(stats, per_90_stats):
        player_matches_df[per_90_stat] = player_matches_df[stat].astype(float) / player_matches_df["time"].astype(int) * 90

    # Only date plus stats
    stats_df = player_matches_df[["date"] + per_90_stats].copy()
    
    # Set index to id and date
    stats_df = stats_df.set_index("date")
        
    return stats_df

def get_multiple_players_stats_df(understat: UnderstatClient, player_ids: List[str],
                      stats: List[str]) -> pd.DataFrame:
    """
    Produces a dataframe with per-90 stats for every game played by each of the given
    player_ids, for any club or season played. Indexes by player_id and date.
    """
    
    stats_dfs = []
    
    for player_id in player_ids:
        stats_df = get_player_stats_df(understat, player_id, stats)
        # Add id to index
        stats_df["player_id"] = player_id
        stats_df = stats_df.set_index("player_id", append=True)
        # Swap to index by player_id
        stats_df = stats_df.swaplevel()
        stats_dfs.append(stats_df)

    return pd.concat(stats_dfs)

class CustomFootballDataset(Dataset):
    """
    A torch dataset to wrap around a stats dataframe for training a time series
    model.
    
    At a given index, one can get a pairing containing metrics for the last 
    window_size games, along with the current game's metrics as the label.
    """
    
    def __init__(self, stats_df: pd.DataFrame, window_size: int = 10, multiple_players: bool = True):
        """
        Initializes a CustomFootballDataset over the given stats_df, storing model
        inputs and outputs in X and y respectively. Each value in X provides
        a 2d array of stats for the last window_size games, while each value
        in y provides stats for the current game to predict with the matching
        X values.
        """
        super().__init__()
        
        self.X = []
        self.y = []
        
        if multiple_players:   
            # Break down by player
            for _, player_df in stats_df.groupby("player_id"):
                vals = player_df.values
                for i in range(len(vals) - window_size):
                    self.X.append(vals[i:i + window_size])
                    self.y.append(vals[i + window_size])
                    
        else:
            vals = stats_df.values
            for i in range(len(vals) - window_size):
                self.X.append(vals[i:i + window_size])
                self.y.append(vals[i + window_size])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)