"""
K-Nearest Neighbors (KNN) - Iris Flower Classification
======================================================
Implements KNN algorithm to classify iris flowers using scikit-learn.

Author: Your Name
Date: 2024
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


def load_and_prepare_data(test_size=0.2, random_state=42):
    """
    Load and split the Iris dataset.
    
    Args:
        test_size (float): Proportion of dataset for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_data, test_data, train_target, test_target, dataset)
    """
    # Load Iris dataset
    dataset = load_iris()
    data = dataset.data
    target = dataset.target
    
    # Split into training and testing sets
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=test_size, random_state=random_state
    )
    
    return train_data, test_data, train_target, test_target, dataset


def train_knn_model(train_data, train_target, n_neighbors=5):
    """
    Train a KNN classifier.
    
    Args:
        train_data (ndarray): Training features
        train_target (ndarray): Training labels
        n_neighbors (int): Number of neighbors to use (default: 5)
    
    Returns:
        KNeighborsClassifier: Trained model
    """
    # Initialize and train the KNN algorithm
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(train_data, train_target)
    
    return model


def evaluate_model(model, test_data, test_target, target_names):
    """
    Evaluate the trained model and print metrics.
    
    Args:
        model: Trained classifier
        test_data (ndarray): Test features
        test_target (ndarray): True labels
        target_names (list): Names of target classes
    
    Returns:
        tuple: (predicted_labels, accuracy)
    """
    # Make predictions
    predicted_target = model.predict(test_data)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_target, predicted_target)
    
    return predicted_target, accuracy


def print_results(test_target, predicted_target, accuracy, target_names):
    """
    Print evaluation results in a formatted manner.
    
    Args:
        test_target (ndarray): True labels
        predicted_target (ndarray): Predicted labels
        accuracy (float): Model accuracy
        target_names (list): Names of target classes
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Correct predictions: {np.sum(test_target == predicted_target)}/{len(test_target)}")
    
    print("\n" + "-" * 70)
    print("Sample Predictions:")
    print("-" * 70)
    print(f"{'Index':<8} {'Actual':<20} {'Predicted':<20} {'Correct':<10}")
    print("-" * 70)
    
    for i in range(min(10, len(test_target))):
        actual = target_names[test_target[i]]
        predicted = target_names[predicted_target[i]]
        correct = "✓" if test_target[i] == predicted_target[i] else "✗"
        print(f"{i:<8} {actual:<20} {predicted:<20} {correct:<10}")
    
    print("\n" + "-" * 70)
    print("Confusion Matrix:")
    print("-" * 70)
    cm = confusion_matrix(test_target, predicted_target)
    print(cm)
    
    print("\n" + "-" * 70)
    print("Classification Report:")
    print("-" * 70)
    print(classification_report(test_target, predicted_target, target_names=target_names))


def main():
    """Main execution function."""
    print("=" * 70)
    print("K-Nearest Neighbors (KNN) - Iris Flower Classification")
    print("=" * 70)
    
    # Load and prepare data
    print("\n1. Loading Iris dataset...")
    train_data, test_data, train_target, test_target, dataset = load_and_prepare_data(
        test_size=0.2, random_state=42
    )
    
    print(f"   Dataset shape: {dataset.data.shape}")
    print(f"   Features: {dataset.feature_names}")
    print(f"   Classes: {list(dataset.target_names)}")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Testing samples: {len(test_data)}")
    
    # Train model
    print("\n2. Training KNN model...")
    n_neighbors = 5
    model = train_knn_model(train_data, train_target, n_neighbors=n_neighbors)
    print(f"   Model trained with k={n_neighbors} neighbors")
    
    # Evaluate model
    print("\n3. Evaluating model on test data...")
    predicted_target, accuracy = evaluate_model(
        model, test_data, test_target, dataset.target_names
    )
    
    # Print detailed results
    print_results(test_target, predicted_target, accuracy, dataset.target_names)
    
    print("\n" + "=" * 70)
    print("Classification complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
