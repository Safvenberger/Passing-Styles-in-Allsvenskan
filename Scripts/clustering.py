#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objects as go
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
                      "Defensive Midfield", "Central Midfield", "Attacking Midfield",
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
        columns=all_passing_adjusted_per_90.drop(["player", "team", "cluster"], 
                                                 axis=1).columns)    
    
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
    
    
def fit_player_cluster(pca_data: DataFrame, all_passing_adjusted_per_90: DataFrame,
                       all_passing_adjusted_per_90_scaled: DataFrame,
                       nr_clusters: int) -> DataFrame:
    """
    Perform clustering on the PCA data for players.

    Parameters
    ----------
    pca_data : DataFrame
        The data projected onto the PCA bases.
    all_passing_adjusted_per_90_scaled :
        All passing events, standardized by minutes played, possession adjusted 
        and standardized per 90 minutes.
    all_passing_adjusted_per_90 : DataFrame
        All passing events, standardized by minutes played, and possession adjusted
        per 90 minutes.
    nr_clusters : int
        The number of clusters to consider.

    Returns
    -------
    plot_data : DataFrame
       The PCA with cluster labelled data to be plotted for visual examination.

    """
    
    # Choose a satisfactory value of k for the final clustering
    k_medoids = KMedoids(n_clusters=nr_clusters, random_state=0).fit(pca_data)

    # Get the medoid in each cluster
    medoids = all_passing_adjusted_per_90.iloc[
        k_medoids.medoid_indices_].reset_index().iloc[:, :2]
    
    # Add cluster label
    medoids.insert(0, "Cluster", np.array(range(1, nr_clusters+1))) 
       
    print(medoids)
       
    # Save the clustering labels
    all_passing_adjusted_per_90["cluster"] = k_medoids.labels_+1
    
    # Reset the index to get player and teams
    all_passing_adjusted_per_90.reset_index(inplace=True)
    
    # Evalute the positions within clusters
    position_and_cluster = visualize_position_distribution(all_passing_adjusted_per_90_scaled,
                                                           all_passing_adjusted_per_90)
    
    # Evaluate cluster means and ranks
    visualize_cluster_means(all_passing_adjusted_per_90_scaled,
                            all_passing_adjusted_per_90)
    
    # Create a data frame for plotting
    plot_data = pd.concat([pd.DataFrame(pca_data), 
                           all_passing_adjusted_per_90[["player", "team", "cluster"]]],
                          axis=1).rename(columns={0: "PC1", 1: "PC2", 2: "PC3", 
                                                  3: "PC4", 4: "PC5"})
    
    # Combine performance data with player metadata
    plot_data = plot_data.merge(position_and_cluster.drop(["Height", "Foot"], axis=1),
                                on=["player", "team", "cluster"])
                
    # Transform market value
    plot_data["Market value (M)"] = plot_data["Market value"].str.replace("€", "").apply(
        lambda x: np.nan if isinstance(x, float) else (
            0.001 * float(x[:-3]) if "Th" in x else 1 * float(x[:-1]) ))
           
    # Save the data to a csv file
    plot_data.to_csv("../Data/plot_data.csv", index=False)
    
    return plot_data


def interactive_player_clustering(plot_data: DataFrame, nr_clusters: int) -> None:
    """
    Create an interactive plot for examining the clustering of players

    Parameters
    ----------
    plot_data : DataFrame
        The data used for plotting.

    Returns
    -------
    None. Instead an interactive html plotly figure is created.

    """
           
    # Find the different clusters
    clusters = plot_data["cluster"].sort_values().unique()
    
    # Specify a colorblind friendly palette
    cbPalette = ('#4477AA', '#882255', '#228833', '#EE7733', '#EE3377', '#DDAA33', 
                 '#BBBBBB', '#88CCEE')
    
    # Create as many traces as different groups there are and save them in data list
    data = []
    for cluster in clusters:
                
        # Find all players assigned to a specific cluster
        df_cluster = plot_data[plot_data["cluster"] == cluster]

        # Create the hovertext
        hovertext = df_cluster.apply(
            lambda x: f"""{x.player} ({x.team})<br>Cluster: {cluster}<br>Position: {x.position}""",
            axis=1)
        
        # Add a trace for each cluser so they all get a distinct color
        trace = go.Scatter(x=df_cluster["PC1"], 
                           y=df_cluster["PC2"],
                           mode="markers",
                           name=f"Cluster{cluster}",
                           marker=dict(color=cbPalette[cluster-1]),
                           hovertext=hovertext)
        
        # Add the trace to the data list
        data.append(trace)
    
    # Layout of the plot
    layout = go.Layout(title="", plot_bgcolor="white",
                       xaxis = dict(
                           title="PC1",
                           ticks="inside"
                           ),
                       yaxis = dict(
                           title="PC2",
                           ticks="inside"
                           ),
                       # autosize=False, height=700, width=1080
                       )
    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Plot the figure
    plot(fig, filename="passing-styles.html")


def getImage(path: str):
    """
    From a given path, read an image file.

    Parameters
    ----------
    path : str
        The path to the image.

    Returns
    -------
    An image file, suitable for plotting.

    """
    return OffsetImage(plt.imread(path), zoom=.03, alpha = 1)


def fit_team_cluster(pca_team_data: DataFrame, 
                     team_passing: DataFrame):
    """
    Fit a hierarchical clustering of teams.

    Parameters
    ----------
    pca_team_data : DataFrame
        The data projected onto the PCA bases.
    team_passing : DataFrame
        The passing statistics per team.

    Returns
    -------
    None. Instead, a figure is created and saved.

    """
    
    # Create a hierarchical clustering of teams based on standardized passing statistics
    team_clusters = shc.linkage(pca_team_data, method="ward", 
                                metric="euclidean", optimal_ordering=True)
    
    # Initialize a figure
    fig, ax = plt.subplots(figsize=(12, 8))       
    
    # Change the linewidth of the dendrogram
    with plt.rc_context({"lines.linewidth": 2.5}):
        # Create a dendrogram showing team clustering
        s = shc.dendrogram(Z=team_clusters, orientation="right",
                           labels=team_passing.index)
    
    # Specify axis labels
    ax.set_xlabel("Ward distance", fontsize=14)
    ax.set_title("Hierarchical clustering of team passing statistics")
    
    # Make the y-axis ticks white in color 
    ax.tick_params(axis="y", colors="white", rotation=80)
    
    # Create a data frame to contain the path to each team's badge icon
    team_badge_data = pd.DataFrame(team_passing.index)
    team_badge_data["path"] = team_badge_data.apply(
        lambda x: f"../Badges/{x.team}.png", axis=1)    
    
    # Extract the leaf of each team
    team_leaf = pd.DataFrame.from_dict(
        {team:leaf for team, leaf in zip(s["ivl"], s["leaves"])}, 
    orient="index", columns=["leaf"]).reset_index().rename(
        columns={"index": "team"}).reset_index()
    
    # Combine team badge path and leaf in the plot
    team_badge_data = team_badge_data.merge(team_leaf, on="team").sort_values("index")

    # Get the axis tick position
    team_badge_data["position"] = [i.get_position()[1] for i 
                                   in ax.get_yticklabels()]                
                     
    # Add the team logo for each leaf
    for index, row in team_badge_data.iterrows():
        ab = AnnotationBbox(getImage(row["path"]), (-0.5, row["position"]), 
                            frameon=False, annotation_clip=False)
        ax.add_artist(ab)
    
    # Save figure
    plt.tight_layout()
    plt.savefig("../Figures/hiearchical_cluster_team.png", dpi=300)
        
    # Plot the figure    
    plt.show()

