#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt   
from typing import Tuple
from pandera.typing import DataFrame
import warnings

# Suppress numpy warning
warnings.filterwarnings(action="ignore", category=np.ComplexWarning)

def visualize_pca(passing_scaled: DataFrame,
                  passing: DataFrame,
                  team: bool=False) -> None:
    """
    Visualize the explained variance obtained from PCA.

    Parameters
    ----------
    passing_scaled : DataFrame
        All passing events, standardized by minutes played (if player), 
        possession adjusted and standardized per 90 minutes.
    passing : DataFrame
        All passing events, standardized by minutes played (if player), 
        and possession adjusted per 90 minutes.
    team : bool
        If the passing statistics should be per team (True) or player (False).
        
    Returns
    -------
    None. Instead a plot is created.

    """
    
    # Specify the figure suffix
    fig_suffix = ""

    # Figure suffix for team
    if team:
        fig_suffix = "_team"
        
    # Create a PCA object for selecting the components
    pca = PCA().fit(passing_scaled)

    # Find the number of components
    nr_components = np.amin((len(passing.columns),
                             len(passing)))
    
    # Initialize figure
    plt.figure(figsize=(12,8))
    
    # Create a data frame for plotting
    explained_variance = pd.DataFrame(
        data=np.around(pca.explained_variance_ratio_, 3),
        index=[f'PC{i}' for i in range(1, nr_components+1)],
        columns=["ExplainedVariance"]).reset_index().rename(columns={"index": "PC"})
    
    # Compute the cumulative explained variance
    explained_variance["CumulativeExplained"] = np.cumsum(explained_variance["ExplainedVariance"])
    
    # Create a barplot for the explained variance for each component
    ax = sns.barplot(x="PC", y="ExplainedVariance",
                     data=explained_variance, 
                     color="#E8112d")
    
    # Create a scatterplot between the components and the cumulative explained variance
    sns.scatterplot(x="PC", y="CumulativeExplained",
                    data=explained_variance, alpha=0.7,
                    color="#006AB3")
    
    # Create a lineplot for the cumulative explained variance of the components
    sns.lineplot(x="PC", y="CumulativeExplained",
                 data=explained_variance, alpha=0.7,
                 color="#006AB3")
    
    # Remove whitespace between bars and plot axes
    ax.margins(x=0)
    
    # Specify axis ticks on the y axis
    ax.set_yticks(ticks=np.linspace(0,1,11))
    
    # Specfiy axis text size
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=12)

    # Specify axis labels
    ax.set_xlabel("Principal component", fontsize=14)
    ax.set_ylabel("Proportion of variance explained", fontsize=14)
    
    # Specify the boundaries of the y axis
    ax.set_ylim(0, 1)

    # Save figure
    plt.tight_layout()
    plt.savefig(f"../Figures/pca_explained{fig_suffix}.png", dpi=300)
    

def visualize_pca_loadings(passing_scaled: DataFrame, 
                           passing: DataFrame,
                           team: bool=False) -> None:
    """
    Visualize the loadings of each variable on the principal components

    Parameters
    ----------
    passing_scaled : DataFrame
        All passing events, standardized by minutes played (if player), 
        possession adjusted and standardized per 90 minutes.
    passing : DataFrame
        All passing events, standardized by minutes played (if player), 
        and possession adjusted per 90 minutes.
    team : bool
        If the passing statistics should be per team (True) or player (False).
        
    Returns
    -------
    None. Instead a plot is created.

    """
    
    # Specify the figure suffix
    fig_suffix = ""

    # Figure suffix for team
    if team:
        fig_suffix = "_team"
        
    # Create a new PCA object with the desired amount of components
    pca = PCA().fit(passing_scaled)
    
    # Find the number of components
    nr_components = np.amin((len(passing.columns),
                             len(passing)))
    
    # Create a data frame containing the PC loadings
    loadings = pd.DataFrame(
       data=pca.components_.T * np.sqrt(pca.explained_variance_), 
       columns=[f'PC{i}' for i in range(1, nr_components+1)],
       index=passing.columns
     )   

    # Initialize figure
    plt.figure(figsize=(12, 8))    
    
    # Plot the heatmap of loadings    
    ax = sns.heatmap(loadings, cmap="RdYlGn", vmin=-1, vmax=1,
                     cbar_kws={"label": "PCA loading"})

    # Specfiy axis text size
    ax.tick_params(axis="both", labelsize=12)

    # Rotate x-axis ticks
    ax.tick_params(axis="x", rotation=90)

    # Get the colorbar
    cbar = ax.collections[0].colorbar
    
    # Specify font size for the colorbar
    cbar.ax.tick_params(labelsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"../Figures/pca_loadings{fig_suffix}.png", dpi=300)
    

def visualize_number_of_clusters(pca_data: DataFrame,
                                 team: bool=False) -> None:
    """
    Visualize the relationship between the number of clusters and cluster 
    tightness via Silhouette score. A higher score indicates more distinct clusters.

    Parameters
    ----------
    pca_data : DataFrame
        The data projected onto the PCA bases.
    team : bool
        If the passing statistics should be per team (True) or player (False).

    Returns
    -------
    None. Instead a plot is created.

    """
    
    # Specify the figure suffix
    fig_suffix = ""

    # Figure suffix for team
    if team:
        fig_suffix = "_team"
        
    # Initialize an empty array for storing silhouette scores
    silhouette = np.zeros((14, 2))
    
    # Examine k = 2, ..., 15
    for k in range(2, 16):
        # Compute PAM on the PCA data
        k_medoids = KMedoids(n_clusters=k, random_state=0).fit(pca_data)
        
        # Save the current value of k in the first column
        silhouette[k-2, 0] = k
        
        # Compute the silhouette score for the current clustering
        silhouette[k-2, 1] = silhouette_score(pca_data, k_medoids.labels_)
        
    # Find the largest silhouette scores    
    best_k = silhouette[:, 1].argsort()[::-1] + 2
    
    # Create a data farme for plotting
    plot_data = pd.DataFrame(silhouette, columns=["k", "Silhouette"])

    # Initialize figure    
    plt.figure(figsize=(12, 8))    

    # Create a barplot showing the best values for k
    ax = sns.barplot(data=plot_data, x="k", y="Silhouette", order=best_k, 
                     color="#E8112d")
    
    # Specify axis labels
    ax.set_xlabel("Number of clusters", fontsize=14)
    ax.set_ylabel("Silhouette score", fontsize=14)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"../Figures/number_of_clusters{fig_suffix}.png", dpi=300)
    
    
def parallel_analysis(input_data: np.ndarray, iterations: int=100, centile: int=99, 
                      plot: bool=True, team: bool=False) -> Tuple[int, np.array]:
    """
    Perform parallel analysis based on Horn (1965) and the improvements 
    suggested by Glorfeld (1995) to select the number of components in PCA.

    Parameters
    ----------
    input_data : np.ndarray
        The data on which the parallel analysis is based.
    iterations : int, optional
        The number of iterations to run for sampling. The default is 100.
    centile : int, optional
        Computes a Monte Carlo estimate for estimating bias. Accepts values 
        between 0 and 99, where 0 corresponds to using the mean.
        The default is 99.
    plot : bool, optional
        If a plot should be created. The default is True.
    team : bool
        If the passing statistics should be per team (True) or player (False).
        
    Returns
    -------
    retained : int
        The number of retained components.
    adj_eigenvalues : np.array
        Array of all adjusted eigenvalues.

    References
    ----------
    Horn, J. L. (1965). A rationale and test for the number of factors in factor
    analysis. Psychometrika, 30(2), 179–185.
    
    Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology 
    for selecting the correct number of factors to retain.
    Educational and Psychological Measurement, 55(3), 377–393. 

    """

    # Specify the figure suffix
    fig_suffix = ""

    # Figure suffix for team
    if team:
        fig_suffix = "_team"
        
    # Set seed for reproducible results
    np.random.seed(0)
    
    # Get the number of observations (n) and variables (p)
    n, p = input_data.shape
    
    # Find the minimum of n and p
    min_dim = np.amin((n, p))
    
    if n < p:
        warnings.warn("The number of observations is less than the number of variables")
    
    # Compute the correlation matrix of variables from the data
    corr_matrix = np.corrcoef(input_data, rowvar=False)
    
    # Compute the eigenvalues
    eigenvalues, _ = np.linalg.eig(corr_matrix)    

    # Array to store simulated eigenvalues
    simulated_eigenvalues = np.zeros((iterations, p))

    # Loop over the set of iterations
    for k in tqdm(range(iterations), desc="Performing parallel analysis..."):
        # Sample random a set of (n x p) observations from a standard normal distribution 
        rnd_normal = np.random.normal(size=(n, p))

        # Compute the correlation matrix of variables from the data
        corr_rnd = np.corrcoef(rnd_normal, rowvar=False)
        
        # Compute the eigenvalues
        eigenvalues_rnd, _ = np.linalg.eig(corr_rnd)
        
        # Save the eigenvalues obtained from the random sampling
        simulated_eigenvalues[k, ] = eigenvalues_rnd

    # If the centile method should be used        
    if centile > 0 and centile < 100:
        rnd_ev = np.quantile(simulated_eigenvalues, centile/100, axis=0)[:min_dim]
    elif centile == 0:
        rnd_ev = np.mean(simulated_eigenvalues, axis=0)[:min_dim]
    else: 
        sys.exit("Centile need to be between 0 and 99.")

    # Keep the first min_dim eigenvalues, in the case n < p 
    eigenvalues = eigenvalues[:min_dim]

    # Compute bias for the random eigenvalues
    bias = rnd_ev - 1
    
    # Compute adjusted eigenvalues based on the bias
    adj_eigenvalues = eigenvalues - bias

    # Count the number of components to retain
    retained = (adj_eigenvalues > 1).sum()

    # To plot or not to plot
    if plot:
        # Initialize figure
        fig, ax = plt.subplots(figsize=(12, 8))       
        
        # Plot the unadjusted eigenvalues
        plt.plot(eigenvalues, "-o", color="tab:blue", label="Unadjusted eigenvalues")
           
        # Plot the adjusted eigenvalues to retain
        plt.plot(adj_eigenvalues[adj_eigenvalues > 1], "-o", color="black", 
                 label="Adjusted eigenvalues (retained)")
        
        # Plot the adjusted eigenvalues to not retain
        plt.plot(adj_eigenvalues, "-o", color="black", 
                 label="Adjusted eigenvalues (unretained)", markerfacecolor="none")
        
        # Plot the random eigenvalues
        plt.plot(rnd_ev, "-o", color="tab:red", label="Random eigenvalues")
        
        # Add a line indicating the decision boundary
        plt.axhline(y=1, color="lightgray", alpha=0.5)
        
        # Add legend
        plt.legend()
        
        # Specify axis labels
        ax.set_xlabel("Number of components", fontsize=14)
        ax.set_ylabel("Eigenvalue", fontsize=14)
        ax.set_title("Parallel Analysis")
    
        # Specify the axis ticks
        ticks = [i for i in range(0, eigenvalues.shape[0]+1, 5)]
        ax.xaxis.set_ticks(ticks)
        
        # Set the axis tick labels to be 1 higher
        ax.set_xticklabels([i + 1 for i in ticks])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"../Figures/parallel_analysis{fig_suffix}.png", dpi=300)
        
        # Show plot
        plt.show()  
        
    return retained, adj_eigenvalues


def fit_pca(passing: DataFrame, 
            team: bool=False) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Fit principal component analysis to the data.

    Parameters
    ----------
    passing :
        All passing events, standardized by minutes played (if player), 
        possession adjusted and standardized per 90 minutes.
    team : bool
        If the passing statistics should be per team (True) or player (False).
        
    Returns
    -------
    pca_data : DataFrame
        The data projected onto the PCA bases.
    all_passing_adjusted_per_90 : DataFrame
        All passing events, standardized by minutes played, and possession adjusted
        per 90 minutes.

    """
    
    # Standardize the data by subtracting the mean and dividing by variance
    passing_scaled = scale(passing)
    
    # Visualize the principal components
    visualize_pca(passing_scaled,
                  passing, team=team)

    # Visualize the PCA loadings per component
    visualize_pca_loadings(passing_scaled, 
                           passing, team=team)    
    
    # Use parallel analysis to determine how many components to keep
    retained, _ = parallel_analysis(input_data=passing_scaled, 
                                    iterations=10000, centile=99, plot=True, team=team)

    # Create a new PCA object with the desired amount of components and 
    # project the original data onto the principal components
    pca_data = PCA(n_components=retained).fit_transform(passing_scaled)

    # Visualize which k should be chosen as the number of clusters
    visualize_number_of_clusters(pca_data, team=False)

    return pca_data, passing_scaled
