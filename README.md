# Passing styles in Allsvenskan 2021

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![R](https://img.shields.io/badge/r-%23276DC3.svg?style=for-the-badge&logo=r&logoColor=white)


Have you ever thought about the most common event in football, the pass, and the various ways it is possible to play it? Well, look no further!
In this repository, we conduct an unsupervised learning clustering approach to detect the various passing styles prevalent in Allsvenskan during the 2021 season.

For a detailed post with analysis see [here](https://safvenberger.github.io/blog/passing-styles-in-allsvenskan-2021) 

## Prerequisites
In order to replicate the results in repo, you need access to the data from [PlaymakerAI](https://twitter.com/playmakerai) and the extra file *allsvenskan, 2021-positions.csv*, which is located in the *allsvenskan, 2021* folder.

## Description of this repository
This repo contains the following folders:

- *allsvenskan, 2021*: the folder containing all the data from the game .json files
- *Data*: contains output data from the analysis, e.g., the data used for plotting
- *Figures*: storage for all figures that are created throughout the analysis 
- *Scripts*: all scripts used for conducting the analysis. The main script to run is ```passingStyles.py``` as it imports the required functions etc. from the other scripts.

## Acknowledgements
I would like to thank [PlaymakerAI](https://twitter.com/playmakerai) for making their data available.

<img src="https://drive.google.com/uc?id=132cTHOhFloLxc3-2B-qmgQlxGsWN0KS7" width="200" height="200" />
