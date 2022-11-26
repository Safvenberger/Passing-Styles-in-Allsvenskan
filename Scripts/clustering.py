#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objects as go
from typing import List, Tuple
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

    Source: https://fcpython.com/visualisation/creating-scatter-plots-with-club-logos-in-python
    
    Parameters
    ----------
    path : str
        The path to the image.

    Returns
    -------
    An image file, suitable for plotting.

    """
    return OffsetImage(plt.imread(path), zoom=.03, alpha = 1)


def get_team_badges(team_passing: DataFrame, tree: dict, ax) -> DataFrame:
    """
    Create a data frame with the paths to an image of each team's badge.

    Parameters
    ----------
    team_passing : DataFrame
        The passing statistics per team.
    tree : dict
        A dict as returned by shc.dendrogram().
    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib figure axes.

    Returns
    -------
    team_badge_data : DataFrame
        Data frame with information over team name, badge path, and axis position.

    """
     # Create a data frame to contain the path to each team's badge icon
    team_badge_data = pd.DataFrame(team_passing.index)
    team_badge_data["path"] = team_badge_data.apply(
        lambda x: f"../Badges/{x.team}.png", axis=1)    
    
    if tree is not None:
        # Extract the leaf of each team
        team_leaf = pd.DataFrame.from_dict(
            {team:leaf for team, leaf in zip(tree["ivl"], tree["leaves"])}, 
            orient="index", columns=["leaf"]).reset_index().rename(
            columns={"index": "team"}).reset_index()
        
        # Combine team badge path and leaf in the plot
        team_badge_data = team_badge_data.merge(team_leaf, on="team").sort_values("index")
    
        # Get the axis tick position
        team_badge_data["position"] = [i.get_position()[1] for i 
                                       in ax.get_yticklabels()] 
    else:
        # Get the axis tick position
        team_badge_data["position"] = [i.get_position()[0] for i 
                                       in ax.get_xticklabels()] 

    return team_badge_data   


def color_links(link_colors: List, linkage: np.ndarray, n: int, 
                link: Tuple, color: str) -> List:
    """
    Update the color of a given link.

    Parameters
    ----------
    link_colors : List
        A list of colors for each link.
    linkage : np.ndarray
        A hiearchical linkage as returned by shc.linkage().
    n : int
        The length of the input data.
    link : Tuple
        A tuple specifying the link to color.
    color : str
        The color to use for the link.

    Returns
    -------
    link_colors : List
        An updated list of colors.

    """

    # Specify link you want to have highlighted
    link_highlight = link
    
    # Find index in clustering where first two columns are equal to link_highlight. 
    index_highlight = np.where((linkage[:, 0] == link_highlight[0]) * 
                               (linkage[:, 1] == link_highlight[1]))[0][0]
    
    # Index in color_list of desired link is index from clustering + length of points
    link_colors[index_highlight + n] = color
    
    return link_colors


def color_cluster(linkage: np.ndarray, team_passing: DataFrame, 
                  color_list: List, n: int, threshold: float) -> List:
    """
    Color all links belonging to a specific cluster.

    Parameters
    ----------
    linkage : np.ndarray
        A hiearchical linkage as returned by shc.linkage().
    team_passing : DataFrame
        The passing statistics per team.
    color_list : List
        A List of colors to use for coloring the lings.
    n : int
        The length of the input data.
    threshold : float
        The distance used for creating clusters.

    Returns
    -------
    link_colors : list
        A list of colors for each link.

    """
            
    # Create a numpy array of [team, cluster, final standing]    
    team_clusters = np.vstack([team_passing.index.to_numpy(), 
                               shc.fcluster(linkage[:, :4], 
                                            t=threshold, criterion="distance"),
                               np.array([2, 12, 13, 3, 14, 5, 4, 8, 7, 11,
                                         6, 1, 9, 10, 15, 16])]).T
    
    # Initialize the link_colors list with gray
    link_colors = ["#BBBBBB"] * (2 * n - 1)

    # Loop over all unique clusters that have been formed
    for cluster in np.unique(team_clusters[:, 1]):
        # Find the index of all the cluster observations
        team_link_idx = np.where(team_clusters[:, 1] == cluster)[0]
        
        # Create empty set to contain the links
        link_idx_set = set()
        all_nodes = set()

        # Default value
        break_cond=False

        while True:
            # Loop over all link indexes between teams
            for team_link in team_link_idx:
                # Find the row index of the team link
                link_idx = np.where((linkage[:, 0] == team_link) |
                                    (linkage[:, 1] == team_link)
                                    )[0]
                
                # Get the link distance
                link_dist = linkage[link_idx, 2][0]
                
                if link_dist < threshold:
                    # Add the link index to a list
                    link_idx_set.add(link_idx[0])
                else:
                    # Distance too large -> break the loop
                    break_cond=True
            
            # Exit the loop
            if break_cond:
                break
            
            # Add the nodes to the "master" set
            all_nodes = all_nodes.union(link_idx_set)

            # Initialize an empty set for the nodes of the parent
            parent_set = set()
            
            # Loop over all 
            for link_idx in link_idx_set:
                # Get the distance and parent of a given link
                link_dist, link_parent = linkage[link_idx, [2, 4]]
                
                if link_dist < threshold:
                    # Add the link parent to the parent set if the distance is 
                    # less than the required disance
                    parent_set.add(int(link_parent))
            
            # Update the team link set
            team_link_idx = parent_set
                
        # Get the color for the cluster
        link_color = color_list[cluster-1]
        
        # Loop over all link indexes among the nodes
        for link_idx in all_nodes:
            # Update the link color list
            link_colors = color_links(link_colors, linkage=linkage, n=n, 
                                      link=tuple(linkage[link_idx, :2]), 
                                      color=link_color)    

    return link_colors

          
def plot_tree(tree: dict, team_badge_data: DataFrame, pos=None, 
              invert: bool=True, distance_threshold: float=None) -> None:
    """
    Plot a tree or a subset of a given tree.
    
    Source: https://stackoverflow.com/questions/16883412/how-do-i-get-the-subtrees-of-dendrogram-made-by-scipy-cluster-hierarchy

    Parameters
    ----------
    tree : dict
        A dict as returned by shc.dendrogram().
    team_badge_data : DataFrame
        Data frame with information over team name, badge path, and axis position.
    pos : iterable, optional
        If specified, a range of indices to plot. The default is None.
    invert : bool
        If the x and y-axis should be inverted.
    distance_threshold : float
        The distance threshold used for creating clusters.

    Returns
    -------
    None. Instead a set of figures are created.

    """
    
    # Initialize a figure    
    fig, ax = plt.subplots(figsize=(12, 8))       

    # Get the coordinates from the dendrogram
    icoord = np.array(tree["icoord"])
    dcoord = np.array(tree["dcoord"])
    
    # Get the list of colors from the dendrogram
    color_list = np.array(tree["color_list"])
    
    # Find the order of smallest to largest distance
    order = np.take(dcoord.argsort(axis=0), 2, axis=1)
    
    # Select the array elements as per the defined order
    icoord = icoord[order]
    dcoord = dcoord[order]
    color_list = color_list[order]
    
    # Determine the boundaries for the axes 
    xmin, xmax = icoord.min(), icoord.max()
    ymin, ymax = dcoord.min(), dcoord.max()
    
    if distance_threshold is not None:
        if invert:
            ymax = distance_threshold
        else:
            xmax = distance_threshold
        # Specify the figure suffix
        fig_suffix = f"threshold_{distance_threshold}"
    else:
        # Specify the figure suffix
        fig_suffix = ""
    
    # If a subset of the dendrogram is to be plotted
    if pos:
        # Get the coordinates of the subset
        icoord = icoord[pos]
        dcoord = dcoord[pos]
        
        # Get the color(s) of the subset
        color_list = color_list[pos]
        
    # Create the plot
    for xs, ys, color in zip(icoord, dcoord, color_list):
        # If the x and y-axis should be inverted
        if invert:
            ax.plot(ys, xs, color)
        else:
            ax.plot(xs, ys, color)
        
    # Add the team logo for each leaf
    for index, row in team_badge_data.iterrows():
        ab = AnnotationBbox(getImage(row["path"]), (-20, row["position"]), 
                            frameon=False, annotation_clip=False,
                            xycoords=("axes points", "data"))
        ax.add_artist(ab)
    
    # If the axes should be inverted
    if invert:
        # Specify plot limits
        ax.set_ylim(xmin-10, xmax + 0.1*abs(xmax))
        ax.set_xlim(ymin, ymax + 0.1*abs(ymax))
                
        # Make the y-axis ticks white in color 
        ax.tick_params(axis="y", colors="white")
        
        # Specify axis labels
        ax.set_xlabel("Distance", fontsize=14)
        
    else:
        # Specify plot limits
        ax.set_xlim(xmin-10, xmax + 0.1*abs(xmax))
        ax.set_ylim(ymin, ymax + 0.1*abs(ymax))
                        
        # Make the y-axis ticks white in color 
        ax.tick_params(axis="y", colors="white", rotation=80)
        
        # Specify axis labels
        ax.set_ylabel("Distance", fontsize=14)
                
    # Set figure title
    ax.set_title("Hierarchical clustering of team passing statistics")
        
    # Save figure
    plt.tight_layout()
    plt.savefig(f"../Figures/TeamClusterIterative/hiearchical_cluster_team_iteration_{fig_suffix}" +
                f"_{pos[-1]}.png", 
                dpi=300)     
        
    # Show plot
    plt.show()


def fit_team_cluster(pca_team_data: DataFrame, 
                     team_passing: DataFrame,
                     plot_iterative_tree: bool=False,
                     threshold: float=None) -> np.ndarray:
    """
    Fit a hierarchical clustering of teams.

    Parameters
    ----------
    pca_team_data : DataFrame
        The data projected onto the PCA bases.
    team_passing : DataFrame
        The passing statistics per team.
    plot_iterative_tree : bool
        If a series of plots should be made for the "growing" process of the dendrogram.
    threshold : float
        The distance used for creating clusters.
    
    Returns
    -------
    team_linkage : np.ndarray
        A hiearchical linkage as returned by shc.linkage().

    """
    
    # Save the length of the data
    n = len(pca_team_data)
    
    # Create a hierarchical clustering of teams based on standardized passing statistics
    team_linkage = shc.linkage(pca_team_data, method="ward", 
                               metric="euclidean", optimal_ordering=True)
    
    # Add a new column to signify the new links
    team_linkage = np.hstack((team_linkage, 
                              np.array([range(n, 2 * n - 1)]).T))
    
    # Initialize a figure
    fig, ax = plt.subplots(figsize=(12, 8))       
        
    # Specify a list of colors
    color_list = ["#762A83", # Purple
                  "#0077BB", # Blue
                  "#117733", # Green
                  "#EEDD88", # Yellow
                  "#EE7733", # Orange                  
                  "#CC3311", # Red
                  "#33BBEE", # Cyan
                  "#FFAABB", # Pink
                  "#555555", # Dark grey
                  "#DDDDDD", # Pale grey
                  ]

    if threshold is not None:
        # Create the colors for each cluster
        link_colors = color_cluster(team_linkage, team_passing, color_list, n, threshold)
        
        # Save the function for coloring the links
        link_color_func = lambda k: link_colors[k]
        
        # Keep only the links below the threshold
        threshold_linkage = team_linkage[team_linkage[:, 2] < threshold]

        # Specify the figure suffix
        fig_suffix = f"_threshold{threshold}"
    else: 
        # Default colors
        link_color_func = None
        
        # Specify the figure suffix
        fig_suffix = ""
        
        # Set the same linkage array
        threshold_linkage = team_linkage
        
    # Change the linewidth of the dendrogram
    with plt.rc_context({"lines.linewidth": 2.5}):
        # Create a dendrogram showing team clustering
        tree = shc.dendrogram(Z=team_linkage[:, :4], orientation="right",
                              labels=team_passing.index,
                              link_color_func=link_color_func)
    
    # Specify axis labels
    ax.set_xlabel("Distance", fontsize=14)
    ax.set_title("Hierarchical clustering of team passing statistics")
    
    # Make the y-axis ticks white in color 
    ax.tick_params(axis="y", colors="white", rotation=80)
    
    # Get the team badges and their relative path
    team_badge_data = get_team_badges(team_passing, tree, ax)
                     
    # Add the team logo for each leaf
    for index, row in team_badge_data.iterrows():
        ab = AnnotationBbox(getImage(row["path"]), (-20, row["position"]), 
                            frameon=False, annotation_clip=False,
                            xycoords=("axes points", "data"))
        ax.add_artist(ab)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"../Figures/hiearchical_cluster_team{fig_suffix}.png", dpi=300)
        
    # Plot the figure    
    plt.show()
    
    # If a iterative tree is to be grown
    if plot_iterative_tree:
        # Loop over all links
        for i in range(1, len(threshold_linkage)+1):
            # Plot each subtree
            plot_tree(tree, team_badge_data, pos=range(i), invert=True,
                      distance_threshold=threshold)
            
    return team_linkage


def team_passing_heatmap(team_passing: DataFrame, team_passing_raw: DataFrame,
                         linkage: np.ndarray, threshold: float=None, 
                         heatmap_difference: bool=False) -> None:
    """
    Create heatmaps for differences between possession adjusted passing statistics
    and raw counts, as well as for the clusters given by threshold.

    Parameters
    ----------
    team_passing : DataFrame
        The possesion adjusted passing statistics per team.
    team_passing_raw : DataFrame
        The original passing statistics per team.
    linkage : np.ndarray
        A hiearchical linkage as returned by shc.linkage().
    threshold : float
        The distance used for creating clusters.
    heatmap_difference : bool
        If the heatmap should visualize the difference between possesion adjusted
        and raw counts (True) or possession adjusted passing statistics (False).

    Returns
    -------
    None. Instead figures are created and saved.

    """

    if threshold is not None:
        # Specify the figure suffix
        fig_suffix = f"_threshold_{threshold}"
    else: 
        # Specify the figure suffix
        fig_suffix = ""
    
    # Create a numpy array of [team, cluster, final standing]    
    team_clusters = np.vstack([team_passing.index.to_numpy(), 
                               shc.fcluster(linkage[:, :4], 
                                            t=threshold, criterion="distance"),
                               np.array([2, 12, 13, 3, 14, 5, 4, 8, 7, 11,
                                         6, 1, 9, 10, 15, 16])]).T
    
    # Get the order as given by each team's cluster 
    cluster_sorting = np.lexsort((team_clusters[:, 2], team_clusters[:, 1]))
    
    # Compute the difference between posession adjusted and raw counts
    team_passing_diff = team_passing - team_passing_raw
            
    if heatmap_difference:
        data = team_passing_diff.copy()
        fig_name = "../Figures/heatmap_team_passing_diff"
        cbar_label = "Standardized difference of possession adjusted passing and raw counts"
    else:
        data = team_passing.copy()
        fig_name = "../Figures/heatmap_team_passing"
        cbar_label = "Standardized possession adjusted passing"
    
    # Create a data frame for scaled differences 
    scaled_data = pd.DataFrame(MinMaxScaler(feature_range=(-1, 1)).fit_transform(data), 
                               index=team_passing.index, columns=team_passing.columns)
    
    # Find columns per pass length
    long_pass = scaled_data.columns[scaled_data.columns.str.contains("Long pass")].to_list()
    medium_pass = scaled_data.columns[scaled_data.columns.str.contains("Medium pass")].to_list()
    short_pass = scaled_data.columns[scaled_data.columns.str.contains("Short pass")].to_list()
    
    # Reorder columns by pass length
    scaled_data = scaled_data[long_pass + medium_pass + short_pass]
    
    # Change row order by cluster order
    scaled_data = scaled_data.iloc[cluster_sorting]
      
    # Rename columns
    scaled_data.columns = scaled_data.columns.str.replace("_", " - ").str.replace(
        "success", "Success").str.replace("fail", "Fail").str.replace("[A-z]+ pass - ", "", regex=True)
    
    # Initialize a figure
    fig, ax = plt.subplots(figsize=(12, 8))       
        
    # Create heatmap
    sns.heatmap(scaled_data.T, cmap="RdYlGn", vmin=-1, vmax=1,
                cbar_kws={"label": cbar_label})
    
    # Specify axis labels
    ax.set_xlabel("", fontsize=14)
        
    # Specify x-axis ticks to be white (= hidden)
    ax.tick_params(axis="x", colors="white", rotation=15)

    # Get the team badges and their relative path
    team_badge_data = get_team_badges(team_passing, tree=None, ax=ax)  
    
    # Get axis position values
    badge_position = team_badge_data.position.values
    
    # Change row order by cluster order
    team_badge_data = team_badge_data.iloc[cluster_sorting].reset_index(drop=True)
      
    # Update position values
    team_badge_data["position"] = badge_position
    
    # Save cluster assignment in badge information
    team_badge_data["cluster"] = team_clusters[cluster_sorting, 1]
    
    # Add the final position for each team
    team_badge_data["tablePosition"] = team_clusters[cluster_sorting, 2]    
    
    # Add the team logo for each leaf
    for index, row in team_badge_data.iterrows():
        ab = AnnotationBbox(getImage(row["path"]), (row["position"], -20), 
                            frameon=False, annotation_clip=False,
                            xycoords=("data", "axes points"))
        ax.add_artist(ab)
    
        # Add cluster label
        ax.annotate(row.cluster, xy=(row["position"], 1.02),
                    annotation_clip=False, horizontalalignment="center",
                    xycoords=("data", "axes fraction"))
        
        # Add final standing
        ax.annotate(f"#{row.tablePosition}", xy=(row["position"], -0.1),
                    annotation_clip=False, horizontalalignment="center",
                    xycoords=("data", "axes fraction"))
    
    # Add cluster number information
    ax.annotate("Cluster number", xy=(0.5, 1.05), fontweight="bold",
                annotation_clip=False, horizontalalignment="center",
                xycoords=("axes fraction", "axes fraction"))
    
    # Add cluster number information
    ax.annotate("Final position", xy=(-0.09, -0.1), fontweight="bold",
                annotation_clip=False, horizontalalignment="center",
                xycoords=("axes fraction", "axes fraction"))
    
    # Loop over all pass lengths
    for idx, pass_length in zip((0, 0.95, 2.1), ["Short",  "Medium", "Long"]):
        # Add passing length information
        ax.annotate(f"{pass_length} passing", xy=(-0.33, 0.075 + (idx) * 0.33), 
                    fontweight="bold", rotation=90, 
                    annotation_clip=False, horizontalalignment="center",
                    xycoords=("axes fraction", "axes fraction"))
    
    # Draw horizontal lines to signify different pass lengths
    ax.hlines([14, 28], *ax.get_xlim(), colors=["black"])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{fig_name}{fig_suffix}.png", dpi=300)
   