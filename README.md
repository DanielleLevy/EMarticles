# Text Clustering and Analysis

## Introduction

This project is designed to perform clustering on a dataset of articles, visualize the results, and analyze the data with respect to various metrics like log-likelihood, perplexity, and accuracy versus lambda. The project is implemented in Python and uses machine learning techniques for clustering.

## Table of Contents

- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)


## Usage

To run the clustering analysis:
python ex3.py <develop.txt> <topics.txt>
This will execute the clustering algorithm on the dataset provided in develop.txt and use the topics from topics.txt for labeling.

## Features
Clustering algorithm implementation with E.M. (Expectation-Maximization).
Visualization of log-likelihood and perplexity over iterations.
Accuracy measurement against a pre-labeled dataset.
Automatic determination of stopping criteria for the E.M. algorithm.
Dependencies
numpy
pandas
matplotlib
scipy
## Configuration
Configuration options can be adjusted in the config.py file (if it exists). This can include the number of clusters, lambda values, and iteration thresholds.

## Documentation
The REPORT.pdf provides a comprehensive overview of the methodology and the results obtained from running the algorithm.

## Examples
After running the algorithm, you can visualize the clustering output using the provided plots in likli.png and myplot.png.

## Troubleshooting
If you encounter any issues with the execution of the code, please check the following:

Ensure all dependencies are installed.
Check the format of develop.txt and topics.txt.
Verify the correctness of the file paths in the code.
## Contributors
Jonathan Mandl 
Danielle Hodaya Shrem 
