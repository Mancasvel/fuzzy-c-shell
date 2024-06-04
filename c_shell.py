"""
© 2024 Manuel Castillejo Vela. All rights reserved.

This code was developed by Manuel Castillejo Vela with assistance from artificial intelligence tools. 
The use of AI was instrumental in providing suggestions and improvements during the coding process. 
Any resemblance to existing code is purely coincidental.

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be 
held liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial applications, 
and to alter it and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. 
   If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO)

class FuzzyCShell:
    def __init__(self, max_clusters: int = 10, max_iter: int = 100, m: float = 1.5, eps: float = 0.01, distance_metric: str = 'euclidean'):
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.m = m
        self.eps = eps
        self.distance_metric = distance_metric
        self.fitted = False

    def distances(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))
        # Could be used in future implementations
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2), axis=-1)
        # For elipsoidal to work properly we need to give as parametres of the init of the cshell algorithm a and b
        elif self.distance_metric == 'elipsoidal':
            return self.elipsoidal_distance(x1, x2, self.a, self.b)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    @staticmethod
    def elipsoidal_distance(x1: np.ndarray, x2: np.ndarray, a: float, b: float) -> float:
        dx = x1[0] - x2[0]
        dy = x1[1] - x2[1]
        return np.sqrt((dx / a) ** 2 + (dy / b) ** 2)

    def initialize_membership_matrix(self, n_samples: int) -> np.ndarray:
        
        membership_matrix = np.random.rand(self.n_clusters, n_samples)
        # Normalize the membership matrix so that the sum of each column is 1
        return membership_matrix / membership_matrix.sum(axis=0)

    def _compute_centroids_and_radii(self, X: np.ndarray, membership_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
        
        um = membership_matrix ** self.m
        # Calculate centroids using the weighted mean formula
        centroids = np.dot(um, X) / np.sum(um, axis=1, keepdims=True)
        distances = np.array([self.distances(X, centroid) for centroid in centroids])
        # Calculate radii as the weighted standard deviation of distances
        radii = np.sqrt(np.sum(um * (distances ** 2), axis=1) / np.sum(um, axis=1))
        return centroids, radii

    def _update_membership_matrix(self, X: np.ndarray, centroids: np.ndarray, radii: np.ndarray) -> np.ndarray:
        
        distances = np.array([self.distances(X, centroid) for centroid in centroids])
        membership_matrix = np.zeros_like(distances)
        # Update the membership matrix
        for i in range(distances.shape[1]):
            for k in range(distances.shape[0]):
                # Update each element of the membership matrix
                membership_matrix[k, i] = 1 / np.sum((distances[k, i] ** 2 / distances[:, i] ** 2) ** (1 / (self.m - 1)))
        return membership_matrix


    def fit(self, X: np.ndarray):
        # Check if input data is 2D
        if len(X.shape) != 2:
            logging.warning("Input data is incorrect. The CSV must have a 2D structure.")
            return None

        # Check if the model is already fitted
        if self.fitted:
            logging.warning("Model fitting has already been completed.")
            return None

        self.fitted = True
        total_samples = X.shape[0]

        # Placeholder for Sum of Squared Errors (SSE)
        error_sums = []
        possible_clusters = [i for i in range(2, self.max_clusters + 1)]

        for clusters in possible_clusters:
            self.n_clusters = clusters
            self.membership_matrix = self.initialize_membership_matrix(total_samples)
            self.centroids, self.radii = self._compute_centroids_and_radii(X, self.membership_matrix)

            iterate = True
            counter = 0
            while iterate and counter < self.max_iter:
                self.centroids, self.radii = self._compute_centroids_and_radii(X, self.membership_matrix)
                temp_membership = self._update_membership_matrix(X, self.centroids, self.radii)
                if np.allclose(temp_membership, self.membership_matrix, atol=self.eps):
                    iterate = False
                self.membership_matrix = temp_membership
                counter += 1

            error_sums.append(self._compute_sse(X))

        self.optimal_n_clusters = self._elbow_method(error_sums)
        logging.info(f"Optimal number of clusters found: {self.optimal_n_clusters}")

        # Final clustering with optimal number of clusters
        self.n_clusters = self.optimal_n_clusters
        self.membership_matrix = self.initialize_membership_matrix(total_samples)
        self.centroids, self.radii = self._compute_centroids_and_radii(X, self.membership_matrix)

        continue_iterating = True
        iteration_count = 0
        while continue_iterating and iteration_count < self.max_iter:
            self.centroids, self.radii = self._compute_centroids_and_radii(X, self.membership_matrix)
            new_membership = self._update_membership_matrix(X, self.centroids, self.radii)
            if np.allclose(new_membership, self.membership_matrix, atol=self.eps):
                continue_iterating = False
                logging.info(f"Clusters determined after {iteration_count + 1} iterations.")
            self.membership_matrix = new_membership
            iteration_count += 1

        return self



    ##Clustering prediction
    
    def _compute_sse(self, X: np.ndarray) -> float:
        distances = np.array([self.distances(X, centroid) for centroid in self.centroids])
        return np.sum(np.min(distances, axis=0) ** 2)

    def _elbow_method(self, sse: list) -> int:
        smoothed_sse = gaussian_filter1d(sse, sigma=1.0)  # Apply Gaussian smoothing for better work with noise
        diffs = np.diff(smoothed_sse)
        second_diffs = np.diff(diffs)
        elbow_point = np.argmax(second_diffs) + 2  # Use maximum curvature for robustness
        return elbow_point

    def predict(self) -> np.ndarray:
        assert self.fitted, "The model has not been fitted yet."
        return np.argmax(self.membership_matrix, axis=0)

# CSV reader
def read_csv_data(file_path: str) -> np.ndarray:
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append([float(row[0]), float(row[1])])
    return np.array(data)

# Simple User Interface using tk
def generate_clusters():
    max_clusters = int(max_clusters_entry.get())
    model = FuzzyCShell(max_clusters=max_clusters, max_iter=100, m=2, eps=0.01, distance_metric='euclidean')
    model.fit(points)
    hard_labels = model.predict()

    plt.scatter(points[:, 0], points[:, 1], c=hard_labels, label='Data Points')

    for idx, (centroid, radius) in enumerate(zip(model.centroids, model.radii)):
        circle = plt.Circle(centroid, radius, color='r', fill=False, linewidth=2, label=f'Cluster {idx + 1}')
        plt.gca().add_patch(circle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

def open_file():
    global points
    file_path = filedialog.askopenfilename()
    points = read_csv_data(file_path)
    file_label.config(text=f"Loaded file: {file_path}")

app = tk.Tk()
app.title("Cluster Generator")

tk.Label(app, text="Max Clusters:").grid(row=0, column=0)
max_clusters_entry = tk.Entry(app)
max_clusters_entry.grid(row=0, column=1)

load_button = tk.Button(app, text="Load Data", command=open_file)
load_button.grid(row=1, column=0, columnspan=2)

generate_button = tk.Button(app, text="Generate Clusters", command=generate_clusters)
generate_button.grid(row=2, column=0, columnspan=2)

file_label = tk.Label(app, text="No file loaded")
file_label.grid(row=3, column=0, columnspan=2)

app.mainloop()