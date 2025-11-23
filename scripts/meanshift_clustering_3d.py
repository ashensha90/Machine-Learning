"""
MeanShift Clustering - Iris Flower 3D Visualization
===================================================
Implements MeanShift clustering algorithm and visualizes results in 3D space.

Author: Your Name
Date: 2024
"""

from sklearn.datasets import load_iris
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def load_iris_data():
    """
    Load the Iris dataset.
    
    Returns:
        tuple: (data, iris_dataset)
    """
    iris = load_iris()
    data = iris.data
    return data, iris


def apply_meanshift_clustering(data, bandwidth=0.85):
    """
    Apply MeanShift clustering algorithm.
    
    Args:
        data (ndarray): Input data
        bandwidth (float): Bandwidth parameter for MeanShift (default: 0.85)
    
    Returns:
        tuple: (labels, centroids, model)
    """
    # Initialize and fit MeanShift clustering
    model = MeanShift(bandwidth=bandwidth)
    model.fit(data)
    
    labels = model.labels_
    centroids = model.cluster_centers_
    
    return labels, centroids, model


def create_3d_visualization(data, centroids, iris):
    """
    Create 3D scatter plot of Iris data with cluster centroids.
    
    Args:
        data (ndarray): Iris dataset
        centroids (ndarray): Cluster centroids
        iris: Iris dataset object
    """
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Split data by species (for coloring)
    # Iris dataset has 50 samples of each species
    new_data = (
        data[:50, :3],      # Setosa
        data[50:100, :3],   # Versicolor
        data[100:150, :3]   # Virginica
    )
    
    colors = ('red', 'green', 'blue')
    species_names = ('Setosa', 'Versicolor', 'Virginica')
    
    # Plot data points for each species
    for species_data, color, species_name in zip(new_data, colors, species_names):
        sepal_length = species_data[:, 0]
        sepal_width = species_data[:, 1]
        petal_length = species_data[:, 2]
        
        ax.scatter(sepal_length, sepal_width, petal_length, 
                  c=color, label=species_name, alpha=0.6, s=50)
    
    # Plot cluster centroids
    for i, centroid in enumerate(centroids):
        centroid_sl, centroid_sw, centroid_pl = centroid[:3]
        ax.scatter(centroid_sl, centroid_sw, centroid_pl, 
                  color='cyan', marker='x', s=200, linewidths=3,
                  label='Centroid' if i == 0 else '')
    
    # Set labels and title
    ax.set_xlabel('Sepal Length (cm)', fontsize=11, labelpad=10)
    ax.set_ylabel('Sepal Width (cm)', fontsize=11, labelpad=10)
    ax.set_zlabel('Petal Length (cm)', fontsize=11, labelpad=10)
    ax.set_title('MeanShift Clustering - Iris Dataset (3D Visualization)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_clustering_summary(labels, centroids, data):
    """
    Print summary of clustering results.
    
    Args:
        labels (ndarray): Cluster labels
        centroids (ndarray): Cluster centroids
        data (ndarray): Original data
    """
    n_clusters = len(np.unique(labels))
    
    print("\n" + "=" * 70)
    print("CLUSTERING SUMMARY")
    print("=" * 70)
    
    print(f"\nNumber of clusters found: {n_clusters}")
    print(f"Total data points: {len(data)}")
    
    print("\n" + "-" * 70)
    print("Cluster Distribution:")
    print("-" * 70)
    
    for cluster_id in range(n_clusters):
        count = np.sum(labels == cluster_id)
        percentage = (count / len(data)) * 100
        print(f"Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
    
    print("\n" + "-" * 70)
    print("Cluster Centroids:")
    print("-" * 70)
    print(f"{'Cluster':<10} {'Sepal Length':<15} {'Sepal Width':<15} "
          f"{'Petal Length':<15} {'Petal Width':<15}")
    print("-" * 70)
    
    for i, centroid in enumerate(centroids):
        print(f"{i:<10} {centroid[0]:<15.2f} {centroid[1]:<15.2f} "
              f"{centroid[2]:<15.2f} {centroid[3]:<15.2f}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("MeanShift Clustering - Iris Dataset with 3D Visualization")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading Iris dataset...")
    data, iris = load_iris_data()
    print(f"   Dataset shape: {data.shape}")
    print(f"   Features: {iris.feature_names}")
    
    # Apply MeanShift clustering
    print("\n2. Applying MeanShift clustering...")
    bandwidth = 0.85
    labels, centroids, model = apply_meanshift_clustering(data, bandwidth=bandwidth)
    print(f"   Bandwidth parameter: {bandwidth}")
    print(f"   Clusters identified: {len(centroids)}")
    
    # Print summary
    print_clustering_summary(labels, centroids, data)
    
    # Create visualization
    print("\n3. Generating 3D visualization...")
    print("   (Close the plot window to continue)")
    create_3d_visualization(data, centroids, iris)
    
    print("\n" + "=" * 70)
    print("Clustering analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
