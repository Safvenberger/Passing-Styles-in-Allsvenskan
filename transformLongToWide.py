#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg


import pandas as pd
import numpy as np
from itertools import groupby, chain


def get_rle(col: str) -> list:
    """
    Create a run-length encoding for a given column.

    Parameters
    ----------
    col : str
        The column to generate a run-length encoding for.

    Returns
    -------
    rle : list
        A list of repeated items (rle) for each value.

    """
    # Create run-length encoding of col
    rle = [sum(1 for i in g) for k, g in groupby(col)]
    
    # Convert to a list of repeated items to match length of df
    rle = list(chain(*[i * [i] for i in rle]))
    
    return rle


def long_to_wide(events: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original events data from long to wide representation with 
    description of the result in binary form (1 = success, 0 = fail).

    Parameters
    ----------
    events : pd.DataFrame
        The original stream of events represented in long format.

    Returns
    -------
    wide : pd.DataFrame
        The events represented in a wide format with the result of an action
        being either 1 or 0.

    """
    # Summarize the actions occurring at the same game_time and position by player
    wide = events.pivot_table(values='result', 
                              index=['match_id', 
                                     'start_time', 'xpos', 'ypos',
                                     'player', 'team', 
                                     'action_order'], 
                              columns='action', aggfunc=np.size).copy()
    
    # Wide shot from goalkeepers perspective not important here
    wide.drop(["Goalkeeper wide shot"], axis=1, inplace=True)
    
    # Since wide shots and shots always co-incide
    wide.drop(["Wide shot"], axis=1, inplace=True)

    # Goal kicks can also always be considered successful
    wide.loc[wide["Goal kick"].notna(), "Goal kick"] = 1      

    # Remove these passes
    wide.drop(["Key pass attempt"], axis=1, inplace=True)
    
    # Blocked shots also coincide with shots
    wide.drop(["Blocked shot"], axis=1, inplace=True)
    
    # Corners are also passes
    wide.loc[wide["Left corner"].notna(), "Pass"] = wide.loc[wide["Left corner"].notna(), "Left corner"]
    wide.loc[wide["Right corner"].notna(), "Pass"] = wide.loc[wide["Right corner"].notna(), "Right corner"]
    
    # Passes into the box have the same outcome as the pass
    wide.loc[wide["Pass into the box"].notna(), "Pass"] = wide.loc[wide["Pass into the box"].notna(), "Pass into the box"]
    
    # Free kicks not ending up a goal
    wide.loc[wide["Direct free kick shot"].notna(),
             "Direct free kick shot"] = 0
    wide.loc[(wide["Goal"].notna() & wide["Direct free kick shot"].notna()), 
             "Direct free kick shot"] = 1
    
    # Change values for shots that become goals
    wide.loc[(wide["Goal"].notna() & wide["Shot"].isna()), 
             "Shot"] = 1
    
    # Specify body part: default is foot
    wide["body_part"] = "foot"
    
    # If aerial challenge => body part is head
    wide.loc[wide["Air challenge"].notna(), "body_part"] = "head"

    return wide


def merge_wide_events(wide_events: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Merge wide events with the original events to combine their information.

    Parameters
    ----------
    wide_events : pd.DataFrame
        The events represented in a wide format.
    events : pd.DataFrame
        The events represented in a long (original) format.

    Returns
    -------
    merged_wide_events : pd.DataFrame
        A combination of the information about events represented in a wide
        format with (nearly all) information available from the long representation.

    """
    # Create copy to avoid changing in-place
    events = events.copy()
    
    # Create a copy to avoid changing in-place
    wide_copy = wide_events.copy()
    
    # Remove the actions not in the SPADL representation
    non_spadl = wide_copy.drop(["Goal mistakes", "Block", "Goal conceded", 
                                "Mistake"], axis=1)
    
    # Keep only numeric columns
    non_spadl = non_spadl.select_dtypes(['number'])
    
    # Combine all non-NA actions into a list
    wide_copy["action_list"] = non_spadl.notna().dot(non_spadl.columns+',').\
        str.rstrip(',')
    
    
    # Combine the result of all actions (1/0) into a list
    wide_copy["action_result"] = non_spadl.apply(lambda x: ",".\
                                                 join([str(i) for i in 
                                                       x.dropna().to_list()]), 
                                                 axis=1)
    
    # Create a new column with actions to merge from the original data frame
    multiple_actions = wide_copy.action_list.str.contains(",")
    wide_copy.loc[~multiple_actions, "merge_action"] = wide_copy.loc[~multiple_actions, 
                                                                     "action_list"]
    
    # Shots
    wide_copy.loc[(wide_copy.merge_action.isna()) &
                  (wide_copy.action_list.str.contains("Shot")),
                  "merge_action"] = "Shot"
    # Passes
    wide_copy.loc[(wide_copy.merge_action.isna()) & 
                  (wide_copy.action_list.str.contains("Pass")), 
                  "merge_action"] = "Pass"
    # Fouls
    wide_copy.loc[(wide_copy.merge_action.isna()) & 
                  (wide_copy.action_list.str.contains("Foul")), 
                  "merge_action"] = "Foul"
    # Dribbles
    wide_copy.loc[(wide_copy.merge_action.isna()) & 
                  (wide_copy.action_list.str.contains("Dribble")), 
                  "merge_action"] = "Dribble"
    # Tackles
    wide_copy.loc[(wide_copy.merge_action.isna()) & 
                  (wide_copy.action_list.str.contains("Tackle")), 
                  "merge_action"] = "Tackle"
    # Interceptions
    wide_copy.loc[(wide_copy.merge_action.isna()) & 
                  (wide_copy.action_list.str.contains("Interception")), 
                  "merge_action"] = "Interception"
    # Air challenges
    wide_copy.loc[(wide_copy.merge_action.isna()) & 
                  (wide_copy.action_list.str.contains("Air challenge")), 
                  "merge_action"] = "Air challenge"
    
    # Challenges
    wide_copy.loc[(wide_copy.merge_action.isna()) & 
                  (wide_copy.action_list.str.contains("Challenge")), 
                  "merge_action"] = "Challenge"
    
    # Drop NA (for older seasons)
    wide_copy = wide_copy.loc[wide_copy.merge_action.notna()]
    
    # Check that the join can be performed correctly
    assert len(wide_copy.loc[wide_copy.merge_action.isna()]) == 0, \
        "There are still merge_actions of length > 1."

    # Replace all goals with shots (as the result of the shots have already been saved)
    events.loc[events.action.eq("Goal"), "action"] = "Shot"

    # Replace all corners with pass
    events.loc[events.action.isin(["Left corner", "Right corner"]), "action"] = "Pass"

    # Rename actions with keeper saves    
    events.loc[events.result_text.eq("saved") |
               events.action.eq("Goal conceded") |
               events.action.eq("Shot on target saved"), "action"] = "Save"    
    
    # Merge wide_copy and events
    merged_wide_events = wide_copy[["action_list", "action_result", 
                                      "body_part", "merge_action"]].reset_index().\
        merge(events,
              left_on=["match_id", "start_time", "xpos", "ypos",
                       "player", "team", "merge_action", "action_order"], 
              right_on=["match_id", "start_time", "xpos", "ypos",
                        "player", "team", "action", "action_order"],
              how="inner")

    return merged_wide_events

    
def fix_multiple_tagged_events(merged_wide_events: pd.DataFrame, event: str,
                               also_contains=",", remove=False) ->  pd.DataFrame:
    """
    Remove multiply tagged events (comma separated strings) to only keep
    the most important action.

    Parameters
    ----------
    merged_wide_events : pd.DataFrame
        Data frame of the merged wide events.
    event : str
        The name of the event to consider.
    also_contains : str, default is ",".
        The additional constraint to consider. 
    remove : str, default is False.
        Whether to remove the given event or not. 

    Returns
    -------
    merged_df : pd.DataFrame
        A modified data frame with only the main action being kept as the
        characterization of the action.

    """
    # Create a copy to avoid changing in-place.
    merged_df = merged_wide_events.copy()
    
    # Replace all events prior to and after the given event
    regex_replace_prior_after = f"([A-z]*)(?={event})|(?<={event})([A-z]*)|into the box"
    
    # Actions with one space
    regex_one_space = f"([A-z]*\s[A-z]*)(?={event})|(?<={event})([A-z]*\s[A-z]*)"
    
    # Actions with two spaces
    regex_two_spaces = f"([A-z]*\s[A-z]*\s[A-z]*)(?={event})|(?<={event})([A-z]*\s[A-z]*\s[A-z]*)"
    
    # Actions with three spaces
    regex_three_spaces = f"([A-z]*\s[A-z]*\s[A-z]*\s[A-z]*)(?={event})|(?<={event})([A-z]*\s[A-z]*\s[A-z]*\s[A-z]*)"
    
    # The string describing which text to be replaced.    
    regex_replace_str = f"{regex_one_space}|{regex_two_spaces}|{regex_three_spaces}|{regex_replace_prior_after}"
    
    # Special cases
    if event == "Pass" and also_contains != ",":
        regex_replace_str = "([A-z]*)(?=Right|Free )|(?<=corner)([A-z]*)|(?<=Cross)([A-z]*)|(?<=kick)([A-z]*)|(?<=throw)([A-z]*)|Pass into the box|Key pass"
    elif event == "Shot" and also_contains != ",":
        regex_replace_str = "(?<=kick shot)([A-z]*)|(?<=Penalty)([A-z]*)|Goal|on target"
    
    # If we should simply remove the given event
    if remove:
        regex_replace_str = f"{event}"
    
    # Find all the rows with the given event and also_contains
    event_index = (merged_df.action_list.str.contains(f"{event}", regex=True)) & \
                  (merged_df.action_list.str.contains(f"{also_contains}", regex=True))
    
    # Replacement values, as specified by regex_replace_str
    if remove:
        replacement = merged_df.loc[event_index, "action_list"].str.\
            replace(regex_replace_str, "", regex=True).str.replace(",$|^,", "", regex=True)
    else:
        replacement = merged_df.loc[event_index, "action_list"].str.\
            replace(",", "", regex=True).str.replace(regex_replace_str, "", regex=True)
    
    # If the replacement is empty and it should be a pass
    replacement.loc[replacement.eq("") & 
                    (event == "Pass") &
                    (also_contains == ",")] = "Pass"
    
    # Replace the list of values with the value to keep
    merged_df.loc[event_index, "action_list"] = replacement        
    
    return merged_df

