#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg


import pandas as pd
import os
import re
import numpy as np
from typing import Tuple
from pandera.typing import DataFrame
from nameMapping import team_name_mapping
from difflib import get_close_matches


def read_event_stream(league: str, year: int, 
                      only_meta: bool=False) -> Tuple[DataFrame, DataFrame]:
    """
    Read the event data from a given league and year from .json files.

    Parameters
    ----------
    league : str
        The name of the league (Allsvenskan/Superettan/Division1).
    year : int
        The year the season was played in.
    only_meta : bool, default is False.
        If only the metadata from each match should be returned.

    Returns
    -------
    match_data : pd.DataFrame
        Metadata about each match.
    event_stream : pd.DataFrame
        A stream of events during all the matches in a season.

    """

    # Relative path to the data
    path = f"{league}, {year}/"
    
    # List all matches for a given competition
    matches = os.listdir(path)

    # Remove minutes played data
    matches = [file for file in matches if "players" not in file and
               file.endswith(".json")]

    # Empty dictionaries    
    match_dict = {}
    event_dict = {}
    
    # Loop over all games
    for match in matches:
        # Read the json file into a pandas dataframe
        df = pd.read_json(f"{path}/{match}", encoding="utf-8")
        
        # Get the meta-data about the match
        meta_data = df.drop("events", axis=1).iloc[:1].\
            rename(columns={"visitingTeam": "awayTeam"})
        
        # Convert score to a list instead of a string that looks like a list
        meta_data[["homeScore", "awayScore"]] = [int(i) for i in \
                                                 re.sub("\[|\]", "", 
                                                        meta_data.loc[0, "score"]).\
                                                     split(", ")]
        
        # Convert score to a list instead of a string that looks like a list
        meta_data[["homeXG", "awayXG"]] = [float(i) for i in \
                                           re.sub("\[|\]", "", 
                                                  meta_data.loc[0, "xG"]).\
                                               split(", ")]
        
        # Assign a unique match_id to each game in the league
        meta_data = meta_data.assign(year=year).rename(columns={"eid": "match_id"})
        
        # Save meta-data in dictionary
        match_dict[f"{match}".strip(".json")] = meta_data.drop(["score", "xG"], 
                                                               axis=1)
            
        # Convert the nested json file of events into a dataframe of its own
        df_events = pd.json_normalize(df["events"])
        
        # Assign the unique match_id to the game as well as the league
        df_events = df_events.assign(match_id=meta_data.loc[0, "match_id"], 
                                     season_id=meta_data.loc[0, "season_id"],
                                     date=meta_data.loc[0, "date"], 
                                     year=year)
        
        # Save events in dictionary
        event_dict[f"{match}".strip(".json")] = df_events
    
    # Combine all meta-data about matches into one dataframe
    match_data = pd.concat(match_dict).reset_index(drop=True)
    
    # Combine all matches into one dataframe
    event_stream = pd.concat(event_dict).reset_index(drop=True)
            
    return match_data, event_stream


def find_names_to_remap(events: DataFrame) -> DataFrame:
    """
    Find players who have two similar, but different names during the same season.
    E.g., Mirko Colak and Antonio Mirko Colak.

    Parameters
    ----------
    events : DataFrame
        A stream of events during all the matches in a season.

    Returns
    -------
    remap_names : DataFrame
        Data frame with mappings of duplicated (but different) player names.

    """
    # Find all players per team 
    team_players = events.groupby(["team", "player"], as_index=False).size()
 
    # Find the closest name among the team players (except the player himself)
    team_players["close_players"] = team_players.apply(
        lambda x: 
            get_close_matches(x.player, 
                              team_players.loc[team_players.player.ne(x.player) & 
                                               team_players.team.eq(x.team), 
                                               "player"], cutoff=0.7), 
        axis=1)
    
    # Extract the names from the list (if applicable)
    team_players["close_players"] = team_players.apply(
        lambda x: x.close_players[0] if x.close_players else np.nan, axis=1)
        
    # Find the names that should not be remapped (manually)
    names_not_to_switch = team_players.close_players.str.contains(
        "Alexander|Andreas|Isak").fillna(False)
    
    # For the names that should not be remapped, set to nan
    team_players.loc[names_not_to_switch, "close_players"] = np.nan
    
    # Remove duplicates
    remap_names = team_players.loc[team_players.close_players.notna()].copy()
    
    # Add player that is missing
    remap_names.loc[len(remap_names)] = ["Hammarby IF", "Mohammed Aziz Ouattara", 
                                         2, "Ouattara Mohammed Aziz"]
    
    remap_names.loc[len(remap_names)] = ["Hammarby IF", "Ouattara Mohammed Aziz", 
                                         1, "Mohammed Aziz Ouattara"]
    
    # Loop over all players to remove
    for player in remap_names.itertuples():
        # The name of the first player
        first_name = player.player
        
        # The name that is proposed to replace the first name
        second_name = player.close_players
        
        # The number of events the first player is tagged with
        size_first_name = player.size
        
        # The number of events the proposed player is tagged with
        size_second_name = remap_names.loc[remap_names.player.eq(second_name), 
                                           "size"].values[0]
        
        # The name which has the most tagged events is the new name
        new_name = first_name if size_first_name > size_second_name else second_name
        
        # Save in the original dataframe
        remap_names.loc[player.Index, "new_name"] = new_name
    
    # Drop duplicated names for faster processing afterwards
    remap_names = remap_names.loc[remap_names.player.ne(remap_names.new_name)].copy()
    
    return remap_names


def sync_player_names(remap_names: DataFrame, df_to_remp: DataFrame) -> DataFrame:
    """
    Synchronize player names in a given data frame.

    Parameters
    ----------
    remap_names : DataFrame
        Data frame with mappings of duplicated (but different) player names.
    df_to_remp : DataFrame
        The data frame to have player names remapped.

    Returns
    -------
    df_copy : DataFrame
        A copy of the original data frame with names remapped.

    """
    # Copy to avoid changing in-place
    df_copy = df_to_remp.copy()
    
    # Loop over all the names that should be remapped to another name
    for player in remap_names.itertuples():
        # Replace the value in the original event data 
        df_copy.loc[df_copy.team.eq(player.team) & 
                    df_copy.player.eq(player.player), "player"] = player.new_name
   
    return df_copy
    

def compute_minutes_played(league: str, year: int, remap_names: DataFrame,
                           min_minutes_played: int=300) -> DataFrame:
    """
    Compute the minutes played for all players during an entire season for a 
    specific league.

    Parameters
    ----------
    league : str
        The league to consider, currently "Allsvenskan".
    year : int
        The year to consider, currently 2021.
    remap_names : DataFrame
        Data frame with mappings of duplicated (but different) player names.
    min_minutes_played : int
        The minimum number of minutes played to be retained in the data.

    Returns
    -------
    minutes_played_season: DataFrame
        Data frame with the total amount of minutes played during the season 
        for all players.

    """       
    # Load minutes played
    minutes = pd.read_json(f"{league}, {year}/{league}, {year}-players.json")
    
    # Rename columns
    minutes.rename(columns={"name": "player"}, inplace=True)

    # Sync player names
    minutes = sync_player_names(remap_names, minutes)

    # Remap team names to match with those from Transfermarkt
    minutes["team"] = minutes.team.map(team_name_mapping())
    
    # Normalize the minutes played from a json file
    minutes_games = pd.json_normalize(minutes["minutes"])
    
    # Get all season ids
    season_ids = pd.Series(minutes_games.columns.str.split(".")).apply(lambda x: x[0])
    
    # Get the most common season  (= the most probable one)
    season_id = season_ids.value_counts().index[0]
    
    # Find games from other seasons than 2021
    other_seasons = ~minutes_games.columns.str.startswith(season_id)
    
    # Remove games from other seasons that may be present in the data
    minutes_games = minutes_games.loc[:, ~other_seasons]
    
    # Remove the json prefix from column names
    minutes_games.columns = minutes_games.columns.str.replace(f"{season_id}.", 
                                                              "", regex=True)
    
    # Combine the meta data and minutes played per game
    minutes_played = pd.concat([minutes[["player", "team"]], minutes_games],
                               axis=1)
     
    # Transform the data from wide to long, where the value is given by minutes played
    minutes_played_long = minutes_played.melt(id_vars=["player", "team"], 
                                              var_name="match_id",
                                              value_name="minutes").dropna()
    
    # Sum the amount of minutes played by each player and team
    minutes_played_season = minutes_played_long.groupby(
        ["player", "team"], as_index=False)["minutes"].sum()
    
    # Remove players who played less than a set number of minutes on the season
    minutes_played_season = minutes_played_season.loc[
        minutes_played_season.minutes.ge(min_minutes_played)]
    
    return minutes_played_season
    
    

if __name__ == "__main__":
    # Input
    league = "allsvenskan"
    year = 2021
    
    # Get the data from the event stream
    match_data, events = read_event_stream(league, year)
        
    # Re-order columns
    events = events[["match_id", "external_id", "game_time", 
                     "start_time", "end_time", "xpos", "ypos", "xdest", "ydest",
                     "player", "team", "action", "next_player", "foot_used",  
                     "header", "throw_in", "penalty", "attack_type", "corner_outcome", 
                     "one_touch", "goal_mouth", "xg", "xp", "xt", "date"]]
    
    # Remap team names to match with those from Transfermarkt
    events["team"] = events.team.map(team_name_mapping())
    
    # Find the player names to remap 
    remap_names = find_names_to_remap(events)
    
    # Remap duplicated player names
    events = sync_player_names(remap_names, events)
    
    # Read data regarding minutes played
    minutes_played_season = compute_minutes_played(league, year, remap_names)
    


    