#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

from preprocess import preprocess_data
from passingStats import get_passing_data
from pca import fit_pca
from clustering import fit_player_cluster, interactive_player_clustering, \
    fit_team_cluster


if __name__ == "__main__":
    
    # Specify input
    league = "Allsvenskan"
    year = 2021
        
    # Read and preprocess the data
    merged_wide_events, minutes_played_season = preprocess_data(league=league,
                                                                year=year)
    
    # Get the passing data for players, adjusted per possession and time played  
    player_passing_per_90 = get_passing_data(merged_wide_events, minutes_played_season,
                                             team=False)
    
    # Get the passing data for teams 
    team_passing = get_passing_data(merged_wide_events, minutes_played_season, 
                                    team=True)
        
    # Fit the PCA for players
    pca_player_data, player_passing_per_90_scaled = fit_pca(player_passing_per_90,
                                                            team=False)
    
    # Fit the PCA for team
    pca_team_data, team_passing_scaled = fit_pca(team_passing,
                                                 team=True)
    
    # Fit a k-mediods clustering of players and get the data for plotting
    plot_data = fit_player_cluster(pca_player_data, player_passing_per_90, 
                                   player_passing_per_90_scaled, nr_clusters=8)
    
    # Create an interactive plot to examine player clustering
    interactive_player_clustering(plot_data)

    # Create a hierarchical clustering of teams
    fit_team_cluster(pca_team_data, team_passing)
     
    
