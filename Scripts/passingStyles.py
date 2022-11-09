#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale 
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from preprocess import preprocess_data
from passingStats import compute_passing_statistics, compute_possession_adjusted_stats,\
    combine_possession_adjusted_and_minutes
from pca import visualize_pca, visualize_pca_loadings, visualize_number_of_clusters, \
    parallel_analysis
from plotly.offline import plot
import plotly.graph_objects as go
from visualizeClustering import visualize_position_distribution, \
    visualize_cluster_means

if __name__ == "__main__":
    
    # Read and preprocess the data
    merged_wide_events, minutes_played_season = preprocess_data(league="Allsvenskan",
                                                                year=2021)
    
    # Compute a pre-defined set of passing statistics to evalute players
    passing_stats = compute_passing_statistics(merged_wide_events)
    
    # Compute possession adjusted passing statistics
    passing_stats_possession_adjusted, corner_and_clearances_possession_adjusted = (
        compute_possession_adjusted_stats(merged_wide_events, passing_stats))
    
    # Standardize passing statistics to be per 90'
    all_passing_adjusted_per_90 = combine_possession_adjusted_and_minutes(
        passing_stats_possession_adjusted, 
        corner_and_clearances_possession_adjusted=None,
        minutes_played_season=minutes_played_season)
    
    # Reorder columns by alphabetical order
    all_passing_adjusted_per_90 = all_passing_adjusted_per_90.reindex(
        sorted(all_passing_adjusted_per_90.columns), axis=1)
    
    # Standardize the data by subtracting the mean and dividing by variance
    all_passing_adjusted_per_90_scaled = scale(all_passing_adjusted_per_90)
    
    # Visualize the principal components
    visualize_pca(all_passing_adjusted_per_90_scaled,
                  all_passing_adjusted_per_90)

    # Visualize the PCA loadings per component
    visualize_pca_loadings(all_passing_adjusted_per_90_scaled, 
                           all_passing_adjusted_per_90)    

    # Use parallel analysis to determine how many components to keep
    retained, _ = parallel_analysis(input_data=all_passing_adjusted_per_90_scaled, 
                                    iterations=10000, centile=99, plot=True)

    # Create a new PCA object with the desired amount of components and 
    # project the original data onto the principal components
    pca_data = PCA(n_components=retained).fit_transform(all_passing_adjusted_per_90_scaled)
    
    # Visualize which k should be chosen as the number of clusters
    visualize_number_of_clusters(pca_data)

    # Choose a satisfactory value of k for the final clustering
    k_medoids = KMedoids(n_clusters=8, random_state=0).fit(pca_data)

    # Get the medoid in each cluster
    medoids = all_passing_adjusted_per_90.iloc[
        k_medoids.medoid_indices_].reset_index().iloc[:, :2]
    
    # Add cluster label
    medoids.insert(0, "Cluster", np.array(range(1, 9))) 
       
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
        
        # Adjust the text position
        # trace.update_traces(textposition="top center")
        
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
                       )
    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Plot the figure
    plot(fig)
