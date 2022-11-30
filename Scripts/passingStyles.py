#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

from preprocess import preprocess_data
from passingStats import get_passing_data
from pca import fit_pca
from clustering import fit_player_cluster, interactive_player_clustering, \
    fit_team_cluster, team_passing_heatmap, team_pca_plot
from matplotlib import font_manager
from matplotlib import rcParams


if __name__ == "__main__":
    
    # Find all fonts
    font_files = font_manager.findSystemFonts(fontpaths="../Fonts")
    
    # Add all fonts
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    
    # Set Roboto as default font
    rcParams["font.family"] = "Roboto"
    
    # Specify input
    league = "Allsvenskan"
    year = 2021
        
    # Read and preprocess the data
    merged_wide_events, minutes_played_season = preprocess_data(league=league,
                                                                year=year)
    
    # Get the passing data for players, adjusted per possession and time played  
    player_passing_per_90 = get_passing_data(merged_wide_events, minutes_played_season,
                                             team=False, possession_adjust=True)
    
    # Get the passing data for teams 
    team_passing = get_passing_data(merged_wide_events, minutes_played_season, 
                                    team=True, possession_adjust=True)
    
    # Get passing data for teams that is not possession adjusted
    team_passing_raw = get_passing_data(merged_wide_events, minutes_played_season, 
                                    team=True, possession_adjust=False)
        
    # Fit the PCA for players
    pca_player_data, player_passing_per_90_scaled = fit_pca(player_passing_per_90,
                                                            team=False)
    
    # Fit the PCA for teams
    pca_team_data, team_passing_scaled = fit_pca(team_passing,
                                                 team=True)
    
    # Fit a k-mediods clustering of players and get the data for plotting
    plot_data = fit_player_cluster(pca_player_data, player_passing_per_90, 
                                   player_passing_per_90_scaled, nr_clusters=8)
    
    # Create an interactive plot to examine player clustering
    interactive_player_clustering(plot_data)

    # Create a hierarchical clustering of teams
    team_linkage = fit_team_cluster(pca_team_data, team_passing, plot_iterative_tree=False)
    
    # Create a hierarchical clustering of teams for a subset
    fit_team_cluster(pca_team_data, team_passing, plot_iterative_tree=True,
                     threshold=5)
    
    # Plot a heatmap of passing differences between possession adjusted and raw counts
    team_passing_heatmap(team_passing, team_passing_raw, linkage=team_linkage,
                         threshold=5, heatmap_difference=True)
    
    # Plot a heatmap of possession adjusted adjusted passing statistics
    team_passing_heatmap(team_passing, team_passing_raw, linkage=team_linkage,
                         threshold=5, heatmap_difference=False)
    
    # Create a scatterplot of team passing statistics in PCA space
    team_pca_plot(pca_team_data, team_passing)
    