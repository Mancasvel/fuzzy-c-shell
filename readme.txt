# Fuzzy C-Shell Clustering Algorithm

This repository contains an implementation of the Fuzzy C-Shell clustering algorithm, along with a test data generator to create synthetic datasets for experimentation. The Fuzzy C-Shell algorithm is an unsupervised clustering method that incorporates fuzzy membership values, allowing data points to belong to multiple clusters with varying degrees of membership.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Fuzzy C-Shell Algorithm](#running-the-fuzzy-c-shell-algorithm)
  - [Generating Test Data](#generating-test-data)
  - [Algorithm structure](#fuzzy-c-shell-clustering-algorithm)
- [License](#license)

## Installation

To get started, clone the repository and install the necessary dependencies using pip:

```bash
git clone https://github.com/mancasvel/fuzzy-c-shell.git
cd fuzzy-c-shell
pip install -r requirements.txt
```

## Usage

### Running the Fuzzy C-Shell Algorithm

The `FuzzyCShell` class implements the Fuzzy C-Shell clustering algorithm. Here is an example of how to use it:

```python
import numpy as np
from fuzzy_c_shell import FuzzyCShell

# Load your data (assuming a 2D numpy array)
You could select the data once you start the application inside your file explorer.

In the case you need to generate data you could run the test_data.py archive.




```

### Generating Test Data

The repository includes a test data generator to create synthetic datasets with specified characteristics. Here is how to generate and save test data:

```python
from test_data import generate_test_data

# Generate and save test data to a CSV file
generate_test_data(
    num_circles=3,
    points_per_circle=60,
    num_noise_points=10,
    noise_radius=30,
    file_path='input_data/example_data.csv'
)
```

This function generates data points arranged in circular clusters with optional noise points and saves the dataset to a CSV file.


# Fuzzy C-Shell Clustering Algorithm

## Overview
The Fuzzy C-Shell Clustering Algorithm is an advanced clustering technique that allows data points to belong to multiple clusters with varying degrees of membership. This implementation is designed to handle 2D datasets and includes several methods for initialization, updating, and determining the optimal number of clusters using the elbow method.

## Code Structure

### Main Classes and Methods

1. **Class: FuzzyCShell**
   - This class implements the Fuzzy C-Shell clustering algorithm.

   - **Method: `fit(self, X: np.ndarray)`**
     - Fits the model to the data `X` by iterating over different numbers of clusters and determining the optimal number using the elbow method.

   - **Method: `initialize_membership_matrix(self, n_samples: int) -> np.ndarray`**
     - Initializes the membership matrix randomly and normalizes it.

   - **Method: `_compute_centroids_and_radii(self, X: np.ndarray, membership_matrix: np.ndarray) -> (np.ndarray, np.ndarray)`**
     - Computes the centroids and radii of the clusters based on the membership matrix.

   - **Method: `_update_membership_matrix(self, X: np.ndarray, centroids: np.ndarray, radii: np.ndarray) -> np.ndarray`**
     - Updates the membership matrix based on the distances between data points and centroids, considering the fuzziness parameter `m`.

   - **Method: `_compute_sse(self, X: np.ndarray) -> float`**
     - Computes the Sum of Squared Errors (SSE) for the current clustering.

   - **Method: `_elbow_method(self, sse: list) -> int`**
     - Determines the optimal number of clusters using the elbow method, enhanced with Gaussian smoothing and maximum curvature.

2. **Data Generation Script: `generate_test_data`**
   - Generates synthetic test data comprising several circles (clusters) and some noise points, and saves the data to a CSV file.

   - **Function: `generate_test_data(num_circles=x, points_per_circle=x, num_noise_points=x, noise_radius=x, file_path='input_data/example_data.csv')`**
     - Generates circular clusters and noise, then saves the points and weights to a CSV file.



## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this code for any purpose, including commercial applications. See the [LICENSE](LICENSE) file for details.

---

© 2024 by Manuel Castillejo Vela. Created with assistance from AI technology. This project is open-source and available for public use.
