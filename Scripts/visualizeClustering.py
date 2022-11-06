#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandera.typing import DataFrame


def visualize_position_distribution(all_passing_adjusted_per_90_scaled: DataFrame, 
                                    all_passing_adjusted_per_90: DataFrame,
                                    league: str="Allsvenskan", year: int=2021) -> DataFrame:
    """
    Compute and visualize position proportions within clusters

    Parameters
    ----------
    all_passing_adjusted_per_90_scaled :
        All passing events, standardized by minutes played, possession adjusted 
        and standardized per 90 minutes.
    all_passing_adjusted_per_90 : DataFrame
        All passing events, standardized by minutes played, and possession adjusted
        per 90 minutes.
    league : str, optional
        The name of the league. The default is "Allsvenskan".
    year : int, optional
        The year to consider. The default is 2021.

    Returns
    -------
    position_and_cluster : DataFrame
        Data frame with player position and cluster.

    """    
    # Read position data
    positions = pd.read_csv(f"../{league}, {year}/{league}, {year}-positions.csv",
                            encoding="utf-8", sep=";")
    
    # Merge cluster information with player position 
    position_and_cluster = all_passing_adjusted_per_90[["player", "team", "cluster"]].merge(
        positions, on=["player", "team"], how="inner")
    
    # Compute the number of players from each position in each cluster
    cluster_position_size = position_and_cluster.groupby(
        ["cluster", "position"], as_index=False).size().sort_values(["cluster", "size"], 
                                                                    ascending=[True, False])
                     
    # Combine withing-cluster position size with cluster size
    cluster_position_size = cluster_position_size.merge(
        cluster_position_size.groupby("cluster")["size"].sum().reset_index(name="clusterSize"))
    
    # Combine position size with within-cluster positions size
    cluster_position_size = cluster_position_size.merge(
        cluster_position_size.groupby("position")["size"].sum().reset_index(name="positionSize"))
                      
    # Compute the percentage of each position within a cluster                                              
    cluster_position_size["prop_cluster"] = 100 * cluster_position_size["size"] / cluster_position_size["clusterSize"]

    # Compute the percentage of each cluster by
    cluster_position_size["prop_position"] = 100 * cluster_position_size["size"] / cluster_position_size["positionSize"]

    # Specify the order of the axis in the barplot
    position_order = ["Goalkeeper", "Centre-Back", "Left-Back", "Right-Back",
                      "Defensive Midfield", "Centre Midfield", "Attacking Midfield",
                      "Left Midfield", "Right Midfield", 
                      "Left Winger", "Right Winger", "Second Striker", 
                      "Centre-Forward"]

    # Initialize a figure
    plt.figure(figsize=(12, 8))                                                                    
    
    # Create a grid of plots
    ax = sns.FacetGrid(cluster_position_size, col="cluster",
                       col_wrap=3, hue="position")
    
    # Map a barplot to each cluster for proportion within cluster
    ax.map(sns.barplot, "prop_cluster", "position", order=position_order)
    
    # Specify axis labels
    ax.set_axis_labels(x_var="% of position in cluster", 
                       y_var="")
    
    # Specify limit
    ax.set(xlim=(0, 100))
    
    # Save figure
    plt.tight_layout()
    plt.savefig("../Figures/barplot_cluster_prop.png", dpi=300)
    
    # Initialize a figure
    plt.figure(figsize=(12, 8))                                                                    
    
    # Create a grid of plots
    ax = sns.FacetGrid(cluster_position_size, col="cluster",
                       col_wrap=3, hue="position")
    
    # Map a barplot to each cluster for proportion of position
    ax.map(sns.barplot, "prop_position", "position", order=position_order)
    
    # Specify limit
    ax.set(xlim=(0, 100))
    
    # Specify axis labels
    ax.set_axis_labels(x_var="% of all in position", 
                       y_var="")
    
    # Save figure
    plt.tight_layout()
    plt.savefig("../Figures/barplot_position_prop.png", dpi=300)
    
    return position_and_cluster

    
def visualize_cluster_means(all_passing_adjusted_per_90_scaled: DataFrame, 
                            all_passing_adjusted_per_90: DataFrame) -> None:
    """
    Visualize the cluster means and their ranks.

    Parameters
    ----------
    all_passing_adjusted_per_90_scaled :
        All passing events, standardized by minutes played, possession adjusted 
        and standardized per 90 minutes.
    all_passing_adjusted_per_90 : DataFrame
        All passing events, standardized by minutes played, and possession adjusted
        per 90 minutes.

    Returns
    -------
    None. Instead a plot is created.

    """
    
    # Create a data frame for computing cluster means based on standardized data
    all_passing_adjusted_per_90_scaled_df = pd.DataFrame(
        data=all_passing_adjusted_per_90_scaled, 
        columns=all_passing_adjusted_per_90.drop(["player", "team", 
                                                  "cluster"], axis=1).columns)    
    
    # Add cluster assignments
    all_passing_adjusted_per_90_scaled_df["cluster"] = all_passing_adjusted_per_90["cluster"]
          
    # Compute cluster means across all numeric passing variables
    means = all_passing_adjusted_per_90_scaled_df.groupby("cluster").mean()
    
    # Initialize a figure
    plt.figure(figsize=(12, 8))                                                                    
    
    # Create a heatmap of passing stats per cluster
    ax = sns.heatmap(means.T, cmap="RdYlGn", 
                     cbar_kws={"label": "Standardized passing"})
    
    # Specfiy axis text size
    ax.tick_params(axis="both", labelsize=12)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    
    # Specify font size for the colorbar
    cbar.ax.tick_params(labelsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig("../Figures/heatmap_cluster_means.png", dpi=300)
    
    # Create a rank version of cluster means (1 = best, 7 = worst)
    mean_ranks = means.rank(axis=0, ascending=False)
    
    # Initialize a figure
    plt.figure(figsize=(12, 8))                                                                    
    
    # Create a heatmap of passing stats per cluster
    ax = sns.heatmap(mean_ranks.T, cmap="RdYlGn_r", 
                     cbar_kws={"label": "Rank (lower rank = more passes)"})
    
    # Specfiy axis text size
    ax.tick_params(axis="both", labelsize=12)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    
    # Specify font size for the colorbar
    cbar.ax.tick_params(labelsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig("../Figures/heatmap_cluster_ranks.png", dpi=300)
    
    
def visualize_team_cluster(plot_data: DataFrame) -> None:
    """
    Visualize the number of players from each team in each cluster.

    Parameters
    ----------
    plot_data : DataFrame
        The PCA with cluster labelled data to be plotted for visual examination.

    Returns
    -------
    None. Instead a plot is created.

    """
    # Compute the number of players in each cluster by team
    team_cluster = plot_data.groupby(["team", "cluster"], as_index=False).size(
        ).sort_values(["cluster", "size"], ascending=[True, False])
        
    # Initialize a figure
    plt.figure(figsize=(12, 8))                                                                    

    # Create a plot for the number of players per cluster and team
    ax = sns.catplot(data=team_cluster, x='size', y= "team", hue="team",
                     col="cluster", dodge=False, 
                     kind='bar', col_wrap=3, sharey=False)
    
    
    # Specify axis labels
    ax.set_axis_labels(x_var="Number of players", 
                       y_var="")
    
    # Save figure
    plt.tight_layout()
    plt.savefig("../Figures/barplot_cluster_team.png", dpi=300)
    