import numpy as np
import csv

def generate_test_data(num_circles=3, points_per_circle=60, num_noise_points=10, noise_radius=30, file_path='input_data/example_data.csv'):
    points = []
    weights = []

    # Generation of random rings
    for _ in range(num_circles):
        center = np.random.uniform(low=0, high=100, size=2)
        radius = np.random.uniform(low=10, high=30)
        angles = np.linspace(0, 2 * np.pi, points_per_circle)
        circle_points = np.array([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])
        
        # Addition of little displacement on the circle points
        displacement = np.random.normal(scale=0.5, size=circle_points.shape)
        circle_points += displacement

        points.extend(circle_points)
        weights.extend(np.ones(points_per_circle))

    # Random noise points aggregation
    noise_points = np.random.uniform(low=0, high=100, size=(num_noise_points, 2))
    points.extend(noise_points)
    weights.extend(np.random.uniform(low=0.1, high=0.5, size=num_noise_points))

    # Save points and weights on the csv
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'weight'])
        for point, weight in zip(points, weights):
            writer.writerow([point[0], point[1], weight])

if __name__ == "__main__":
    generate_test_data()
