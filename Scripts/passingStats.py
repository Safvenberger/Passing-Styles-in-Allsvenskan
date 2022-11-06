#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
from pandera.typing import DataFrame
from typing import Tuple


def compute_passing_statistics(merged_wide_events: DataFrame) -> DataFrame:
    """
    Compute a set of pre-defined passing statistics.

    Parameters
    ----------
    merged_wide_events : DataFrame
        Data frame of all events in wide format.

    Returns
    -------
    passing_stats : DataFrame
        Data frame with all passing statistics per game and player.

    """

    # All passes
    passes = merged_wide_events.loc[(
        merged_wide_events.action.isin(["Pass", "Right corner", "Left corner"]) &
        ~merged_wide_events.action_list.str.startswith("Goal")
        )].copy()
    
    # Pass into final third
    pass_into_final_third = passes.loc[passes.xpos.lt(70) & 
                                       passes.xdest.between(70, 105)].groupby(
        ["match_id", "player", "team", "pass_length", "result_text"], as_index=False).size().rename(
        columns={"size": "Final third entries"})
        
    # Compute number of crosses per player, length and result
    crosses = passes.loc[passes.action_list.eq("Cross")].groupby(
        ["match_id", "player", "team", "pass_length", "result_text"], as_index=False).size(
            ).rename(columns={"size": "Crosses"})
   
    # Compute number of wing switching passes per player, length and result
    wing_switch = passes.loc[passes.wing_switch].groupby(
        ["match_id", "player", "team", "pass_length", "wing_switch", "result_text"], as_index=False).size(
            ).rename(columns={"size": "Wing switches"})
    
    # Compute number of key passes per player, length and result
    key_passes = passes.loc[passes.key_pass].groupby(
        ["match_id", "player", "team", "pass_length", "key_pass", "result_text"], as_index=False).size(
            ).rename(columns={"size": "Key passes"})
            
    # Compute number of progressive passes per player, length and result
    passes_into_box = passes.loc[passes.pass_into_box].groupby(
        ["match_id", "player", "team", "pass_length", "pass_into_box", "result_text"], as_index=False).size(
            ).rename(columns={"size": "Passes into the box"})
            
    # Compute number of assists per player, length and result
    assists = passes.loc[passes.assist].groupby(
        ["match_id", "player", "team", "pass_length", "assist", "result_text"], as_index=False).size(
            ).rename(columns={"size": "Assists"})
    
    # Compute number of progressive passes per player, length and result
    progressive_passes = passes.loc[passes.progressive_pass].groupby(
        ["match_id", "player", "team", "pass_length", "progressive_pass", "result_text"], as_index=False).size(
            ).rename(columns={"size": "Progressive passes"})
 
    # Compute total passes per player, length and result
    total_passes = passes.groupby(
        ["match_id", "player", "team", "pass_length", "result_text"], as_index=False).size(
            ).rename(columns={"size": "Passes"})
    
    # Combine all passing types into one data frame
    passing_stats = (
        wing_switch.drop("wing_switch", axis=1).merge(
        key_passes.drop("key_pass", axis=1), how="outer").merge(
        passes_into_box.drop("pass_into_box", axis=1), how="outer").merge(
        crosses, how="outer").merge(
        assists.drop("assist", axis=1), how="outer").merge(
        progressive_passes.drop("progressive_pass", axis=1), how="outer").merge(
        total_passes, how="outer").merge(pass_into_final_third, how="outer")
            ).sort_values(["player", "team"]).reset_index(drop=True)

    return passing_stats


def compute_possession_adjusted_stats(merged_wide_events: DataFrame, 
                                      passing_stats: DataFrame) -> Tuple[DataFrame,
                                                                         DataFrame]:
    """
    Compute possession adjusted stats for passes and corners for each game 
    and player.

    Parameters
    ----------
    merged_wide_events : DataFrame
        Data frame of all events in wide format.
    passing_stats : DataFrame
        Data frame with all passing statistics per game and player.

    Returns
    -------
    passing_stats_possession_adjusted : DataFrame 
        Data frame with all passing statistics per game and player, adjusted 
        by the team possession in the given game.
    corners_possession_adjusted : DataFrame
        Data frame with statistics of corners by game and player, adjusted
        by the team possession in the given game.

    """
    # Get all passes 
    passes = merged_wide_events.loc[merged_wide_events.action.isin(["Pass", 
                                                                    "Left corner",
                                                                    "Right corner"])]
    
    # Compute number of corners per player and result
    corners = passes.loc[passes.action_list.str.contains("corner")].groupby(
        ["match_id", "player", "team", "result_text"], as_index=False).size(
            ).rename(columns={"size": "Corners"})
            
    # Compute number of clearances per player, length and result            
    clearances = merged_wide_events.loc[merged_wide_events.action.eq("Clearance")].groupby(
        ["match_id", "player", "team", "result_text"], as_index=False).size(
            ).rename(columns={"size": "Clearances"})
            
    # Compute the number of passes per game and team
    game_team_passes = passes.groupby(["match_id", "team"], as_index=False).size().rename(
        columns={"size": "team_passes"})
    
    # Compute the number of passes per game
    game_passes = passes.groupby(["match_id"], as_index=False).size().rename(
        columns={"size": "game_passes"})
    
    # Add the total number of passes to the team game pass data frame
    game_team_passes = game_team_passes.merge(game_passes, on="match_id")
    
    # Compute possession as the ratio of team passes and total passes
    game_team_passes["possession"] = game_team_passes.apply(
        lambda x: x.team_passes / x.game_passes, axis=1)
    
    # Combine passing statistics with possession
    passing_stats_possession = passing_stats.merge(game_team_passes.drop(["team_passes",
                                                                          "game_passes"],
                                                                         axis=1),
                                                   on=["match_id", "team"],
                                                   how="inner")
        
    # Create a copy of the data to compute possession adjusted stats
    passing_stats_possession_adjusted = passing_stats_possession.copy()
    
    # Compute possession adjusted passing statistics
    for column in passing_stats_possession.columns[5:-1]:
         passing_stats_possession_adjusted[column] = passing_stats_possession.apply(
            lambda x: 0.5 * x[column] / x.possession, axis=1)
          
    # Combine corners and clearances
    corner_and_clearances = corners.merge(clearances, how="outer").fillna(0)
         
    # Combine corner and clearance statistics with possession
    corner_and_clearances_possession_adjusted = corner_and_clearances.merge(
        game_team_passes.drop(["team_passes", "game_passes"], axis=1),
        on=["match_id", "team"], how="inner")
    
    # Compute possession adjusted corner and clearance statistics
    for column in corner_and_clearances_possession_adjusted.columns[4:-1]:
         corner_and_clearances_possession_adjusted[column] = corner_and_clearances_possession_adjusted.apply(
            lambda x: 0.5 * x[column] / x.possession, axis=1)
         
    # Compute possession adjusted corner statistics over the entire season
    corner_and_clearances_possession_adjusted = corner_and_clearances_possession_adjusted.groupby(
        ["player", "team", "result_text"], 
           as_index=False).sum().drop(["possession", "match_id"], axis=1)
    
    # Compute possession adjusted passing statistics over the entire season
    passing_stats_possession_adjusted = passing_stats_possession_adjusted.groupby(
        ["player", "team", "pass_length", "result_text"], as_index=False).sum().drop(
            ["possession", "match_id"], axis=1)
    
    return passing_stats_possession_adjusted, corner_and_clearances_possession_adjusted
       

def combine_possession_adjusted_and_minutes(passing_stats_possession_adjusted: DataFrame,
                                            corner_and_clearances_possession_adjusted: DataFrame,
                                            minutes_played_season: DataFrame) -> DataFrame:
    """
    Combine possession adjusted passing statistics with minutes played to 
    compute statistics per 90 minutes.

    Parameters
    ----------
    passing_stats_possession_adjusted : DataFrame
        Data frame with all passing statistics per game and player, adjusted 
        by the team possession in the given game.
    corner_and_clearances_possession_adjusted : DataFrame or None
        Data frame with statistics of corners by game and player, adjusted
        by the team possession in the given game.
    minutes_played_season : DataFrame
        Data frame with the total amount of minutes played during the season 
        for all players.

    Returns
    -------
    all_passing_adjusted_per_90 : DataFrame
        All passing events, standardized by minutes played, and possession adjusted
        per 90 minutes.

    """

    # Combine possession adjusted passing statistics with minutes played
    passing_possession_minutes = passing_stats_possession_adjusted.merge(
        minutes_played_season, left_on=["player", "team"], right_on=["player", "team"])
    
    # Passing variables to consider for standardization
    variables = passing_stats_possession_adjusted.columns[4:]
    
    # Compute possession adjusted passing statistics per 90'
    passing_per_90 = [passing_possession_minutes[[column, "minutes"]].apply(
        lambda x: 90 * x / x.minutes, axis=1).drop("minutes", axis=1) for 
        column in variables]  
    
    # Add the standardized metrics and combine with player name and team
    passing_per_90 = pd.concat([passing_possession_minutes.drop(variables, axis=1), 
                                pd.concat(passing_per_90, axis=1)], axis=1)
    
    # Convert passing from long to wide
    passing_adjusted_per_90 = passing_per_90.drop("minutes", axis=1).pivot(
        index=["player", "team"], columns=["pass_length", "result_text"])
    
    # Rename columns by stacking hierarchical indexes
    passing_adjusted_per_90.columns = ['_'.join(col).strip() for col in
                                       passing_adjusted_per_90.columns.values]
    
    # Reset the index to create common columns for merging merging
    passing_adjusted_per_90.reset_index(inplace=True)
    
    if corner_and_clearances_possession_adjusted is not None:
        # Combine possession adjusted corners + clearances with minutes played
        corners_clearances_minutes = corner_and_clearances_possession_adjusted.merge(
            minutes_played_season, left_on=["player", "team"], right_on=["player", "team"])
        
        # Variable for only success/fail
        success_fail_vars = corner_and_clearances_possession_adjusted.columns[3:]
        
        # Convert the possession adjusted number of corners + clearances to be per 90'    
        corners_clearances_per_90 = [corners_clearances_minutes[[column, "minutes"]].apply(
            lambda x: 90 * x / x.minutes, axis=1).drop("minutes", axis=1) for 
            column in success_fail_vars]  
        
        # Add the standardized metrics and combine with player name and team
        corners_clearances_per_90 = pd.concat([corners_clearances_minutes.drop(success_fail_vars, axis=1), 
                                               pd.concat(corners_clearances_per_90, axis=1)], axis=1)
        
        # Convert corners + clearances from long to wide
        corners_clearances_adjusted_per_90 = corners_clearances_per_90.drop("minutes", axis=1).pivot(
            index=["player", "team"], columns=["result_text"]).fillna(0)

        # Rename columns by stacking hierarchical indexes
        corners_clearances_adjusted_per_90.columns = ['_'.join(col).strip() for col in 
                                                  corners_clearances_adjusted_per_90.columns.values]
        
        # Reset the index to create common columns for merging merging
        corners_clearances_adjusted_per_90.reset_index(inplace=True)        
    
        # Combine per 90' possession adjusted passing with corners + clearances
        all_passing_adjusted_per_90 = passing_adjusted_per_90.merge(
            corners_clearances_adjusted_per_90, how="outer").fillna(0)
    else:
        # If there are no corners/clearances etc. to merge with
        all_passing_adjusted_per_90 = passing_adjusted_per_90.fillna(0)
        
    # Change the index back to the previous version prior to merging
    all_passing_adjusted_per_90.set_index(["player", "team"], inplace=True)

    # Compute the column sums 
    col_sums = all_passing_adjusted_per_90.sum(axis=0)
    
    # Drop all columns that only have 0 as a value
    all_passing_adjusted_per_90.drop(col_sums[col_sums.eq(0)].index, 
                                     axis=1, inplace=True)
    
    return all_passing_adjusted_per_90


