#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import numpy as np
from nameMapping import team_name_mapping, action_order_map
from readData import read_event_stream, find_names_to_remap, sync_player_names, \
    compute_minutes_played
from transformLongToWide import fix_multiple_tagged_events, long_to_wide, \
    merge_wide_events
from typing import Tuple
from pandera.typing import DataFrame


def clean_action_text(events: DataFrame) -> DataFrame:
    """
    Adds an additional column representing the result of an action (success/fail)
    and also removes the additional text present in the "action" column to 
    reduce the number of distinct actions.

    Parameters
    ----------
    events : DataFrame
        The event stream data frame with all events.

    Returns
    -------
    event_df : DataFrame
        Modified data frame with additional columns regarding the result of
        the action performed.

    """
    
    # Copy the data frame to not modify in place
    event_df = events.copy()
        
    # Synchronize goal kicks
    event_df["action"] = event_df["action"].str.replace("Goal-kicks", "Goal kicks", 
                                                        regex=True)
    
    # Convert successful actions to 'success' for consistency
    event_df["action"] = event_df["action"].str.\
        replace(" accurate| won| succesful| successful", " success", 
                regex=True)
        
    # Convert unsuccessful actions to 'fail' for consistency
    event_df["action"] = event_df["action"].str.\
        replace(" lost| unsuccessful| unsuccesful| inaccurate| inacurate", 
                " fail", regex=True)
    
    # Extract the outcome  (in parenthesis) of an action where applicable
    event_df["result_text"] = event_df["action"].str.extract("(success|fail)")
    
    # Convert result to binary 1 if success/saved/GK and 0 otherwise
    event_df["result"] = np.where(event_df["result_text"].isin(["success", 
                                                                "saved",
                                                                "Goalkeeper"]), 
                                  1, 0)
    
    # Set these actions to always success (= 1)
    event_df.loc[event_df["action"].isin(["Assist", "Goal", "Dribbling", 
                                          "Goal kick", "Save"]), "result"] = 1
    
    # Remove all text within parenthesis and the parenthesis aswell
    event_df["action"] = event_df["action"].str.replace("\ssuccess|\sfail", 
                                                        "", regex=True)
    
    return event_df


def add_and_adjust_columns(merged_wide_events: DataFrame) -> DataFrame:
    """
    Add new columns to indicate if a pass was an assist, key pass, or into the box, 
    as well as if a shot was on target and if a foul was a yellow/red card. 
    Also adjusts the multi-tagged action converts the coordinates to a 105x68 pitch.

    Parameters
    ----------
    merged_wide_events : DataFrame
        Data frame of all events in wide format.

    Returns
    -------
    mwe_copy : DataFrame
        Modified data frame with columns adjusted and new columns added.

    """
    # Create a copy to avoid changing in-place
    mwe_copy = merged_wide_events.copy()

    # Sort the data according to game and id
    mwe_copy.sort_values(["match_id", "external_id"], inplace=True)
    
    # Create a column to indicate if a pass was of a specific type
    mwe_copy["assist"] = mwe_copy.action_list.str.contains("Assist")
    mwe_copy["key_pass"] = mwe_copy.action_list.str.contains("ey pass")
    mwe_copy["pass_into_box"] = mwe_copy.action_list.str.contains("into the box")
    mwe_copy["on_target"] = mwe_copy.action_list.str.contains("on target")
    
    # Create an indicator if a foul was awarded a yellow/red card
    mwe_copy["yellow_card"] = mwe_copy.action_list.str.contains("Yellow")
    mwe_copy["red_card"] = mwe_copy.action_list.str.contains("Red")

    # Remove actions that coincide with others
    for action in ["Shot on target", "Assist", "Key pass", "Pass into the box",
                   "Yellow card", "Red card"]:
        mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                              action, remove=True)
    
    # Corners/crosses/goal kicks
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Pass", 
                                          "ross|corner|Goal kick|Goalkeeper")
    # Free kick/penalty shots
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Shot", 
                                          "free kick shot")
    # Fouls
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Foul")
    # Shots
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Shot")
    # Passes
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Pass")
    # Tackles
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Tackle")
    # Dribbles
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Dribble")
    # Interceptions
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Interception")
    # Air challenges
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Air challenge")
    # Remove multiple tagged challenges
    mwe_copy = fix_multiple_tagged_events(mwe_copy, 
                                          "Challenge", 
                                          remove=True)
        
    # Add the result text if it is missing
    mwe_copy["result_text"] = np.where(mwe_copy.result.eq(1), 
                                       "success", "fail")
    
    # Transform coordinates from percentages to meters
    for col in ["xpos", "xdest"]:
        mwe_copy[col] = mwe_copy[col] / 100 * 105
        
    for col in ["ypos", "ydest"]:
        mwe_copy[col] = mwe_copy[col] / 100 * 68
    
    return mwe_copy
    

def compute_distances(merged_wide_events: DataFrame) -> DataFrame:
    """
    Compute the distance of an event and the difference in distance to the goal
    before and after the event. Also adds new columns to indicate the length of 
    a pass, if it was a progressive pass, or if the pass switched wings.

    Parameters
    ----------
    merged_wide_events : DataFrame
        Data frame of all events in wide format.

    Returns
    -------
    mwe_copy : DataFrame
        Modified data frame with new columns related to distance and 
        passing lengths.

    """
    # Create a copy to avoid changing in-place
    mwe_copy = merged_wide_events.copy()

    # Compute the meters travelled in the x and y-direction
    pass_distance_x = mwe_copy.xdest - mwe_copy.xpos 
    pass_distance_y = mwe_copy.ydest - mwe_copy.ypos
    
    # Compute the distance of an action, typically a pass
    mwe_copy["distance"] = np.sqrt(pass_distance_x**2 + pass_distance_y**2)
    
    # Starting distance to the goal in meters, both in the x and y-dimension
    dx_start = (105 - mwe_copy["xpos"]).abs().values
    dy_start = (68/2 - mwe_copy["ypos"]).abs().values
    
    # Distance to the goal from the start of the event
    dist_to_goal_start = np.sqrt(dx_start**2 + dy_start**2)
    
    # End distance to the goal in meters, both in the x and y-dimension
    dx_end = (105 - mwe_copy["xdest"]).abs().values
    dy_end = (68/2 - mwe_copy["ydest"]).abs().values
    
    # Distance to the goal from the end of the event
    dist_to_goal_end = np.sqrt(dx_end**2 + dy_end**2)

    # Save the difference in distance of the start and end of the event
    mwe_copy["distance_to_goal"] = dist_to_goal_start - dist_to_goal_end 
    
    # Determine the distance of an event, typically a pass
    mwe_copy["pass_length"] = np.select(
        [mwe_copy["distance"].ge(40), 
         mwe_copy["distance"].le(15),
         mwe_copy["distance"].between(15, 40, inclusive="neither")],
        ["Long pass", "Short pass", "Medium pass"])
    
    # Determine if an event was progressive, following the wyscout definition
    mwe_copy["progressive_pass"] = (
     (mwe_copy.xpos.le(105/2) & 
      mwe_copy.xdest.le(105/2) &
      mwe_copy.distance_to_goal.ge(30)) |
     (mwe_copy.xpos.le(105/2) & 
      mwe_copy.xdest.ge(105/2) &
      mwe_copy.distance_to_goal.ge(15)) |
     (mwe_copy.xpos.ge(105/2) & 
      mwe_copy.xdest.ge(105/2) &
      mwe_copy.distance_to_goal.ge(10))
     )
    
    # Determine if the pass switch wings, covering at least half the pitch in the y-dimension
    mwe_copy["wing_switch"] = (abs(mwe_copy.ypos - mwe_copy.ydest).ge(68/2))
    
    return mwe_copy


def preprocess_data(league: str, year: int) -> Tuple[DataFrame, DataFrame]:
    """
    Preprocess the data for analysis.

    Parameters
    ----------
    league : str
        The name of the league (Allsvenskan/Superettan/Division1).
    year : int
        The year the season was played in.

    Returns
    -------
    merged_wide_events : DataFrame
        Data frame of all events in wide format.
    minutes_played_season : DataFrame
        Data frame with the total amount of minutes played during the season 
        for all players.

    """
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
    
    # Convert the action column to only contain the action itself and 
    # create a new column for the result of the action
    events_clean = clean_action_text(events)
    
    # Create a column containing action order
    events_clean["action_order"] = events_clean.action.map(action_order_map())
    
    # Re-order values
    events_clean.sort_values(["match_id", "start_time", "action_order",
                              "team", "player"], inplace=True)
    
    # Remove duplicated actions
    events_clean.drop_duplicates(["match_id", "start_time",# "end_time",
                                  "player", "xpos", "ypos", "action"], 
                                  inplace=True)
    
    # Convert from long to wide
    wide_events = long_to_wide(events_clean)
    
    # Combine the information from the wide and long data representation
    merged_wide_events = merge_wide_events(wide_events, events_clean)
    
    # Preprocess the data by adjusting actions, adding new columns etc.
    merged_wide_events = add_and_adjust_columns(merged_wide_events)

    # Compute distances of passes and length of passes
    merged_wide_events = compute_distances(merged_wide_events)
    
    return merged_wide_events, minutes_played_season