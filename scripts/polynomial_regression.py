"""
Polynomial Regression for Non-Linear Data
==========================================
Demonstrates polynomial regression using scikit-learn to fit non-linear data.

Author: Ashen Perera
Date: 2024
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def create_synthetic_data():
    """
    Generate synthetic non-linear data following the equation: y = 3x² - 4x + 5
    
    Returns:
        tuple: (x, y) arrays
    """
    x = np.arange(-1, 1, 0.2)
    x = x.reshape(-1, 1)
    y = 3 * np.power(x, 2) - 4 * x + 5
    
    return x, y


def fit_polynomial_regression(x, y, degree=2):
    """
    Fit a polynomial regression model to the data.
    
    Args:
        x (ndarray): Input features
        y (ndarray): Target values
        degree (int): Degree of polynomial (default: 2)
    
    Returns:
        tuple: (fitted_model, polynomial_features, coefficients)
    """
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)
    
    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(x_poly, y)
    
    # Extract coefficients
    a1 = model.coef_[0][0]  # Coefficient of x
    a2 = model.coef_[0][1]  # Coefficient of x²
    a0 = model.intercept_[0]  # Intercept
    
    return model, poly, (a0, a1, a2)


def plot_results(x, y, coefficients):
    """
    Plot original data and fitted polynomial curve.
    
    Args:
        x (ndarray): Original x values
        y (ndarray): Original y values
        coefficients (tuple): (a0, a1, a2) polynomial coefficients
    """
    a0, a1, a2 = coefficients
    
    # Create scatter plot of original data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Original Data', alpha=0.6)
    
    # Generate smooth curve for best fit line
    dummy_x = np.linspace(-1, 1, 100)
    dummy_y = a2 * np.power(dummy_x, 2) + a1 * dummy_x + a0
    
    plt.plot(dummy_x, dummy_y, 'r--', linewidth=2, 
             label=f'Best Fit: y = {a2:.2f}x² + {a1:.2f}x + {a0:.2f}')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Polynomial Regression (Degree 2)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Polynomial Regression Demo")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data (y = 3x² - 4x + 5)...")
    x, y = create_synthetic_data()
    print(f"   Generated {len(x)} data points")
    
    # Fit polynomial regression
    print("\n2. Fitting polynomial regression model...")
    model, poly, coefficients = fit_polynomial_regression(x, y, degree=2)
    a0, a1, a2 = coefficients
    
    print(f"\n3. Model coefficients:")
    print(f"   Intercept (a0): {a0:.4f}")
    print(f"   Coefficient of x (a1): {a1:.4f}")
    print(f"   Coefficient of x² (a2): {a2:.4f}")
    print(f"\n   Fitted equation: y = {a2:.4f}x² + {a1:.4f}x + {a0:.4f}")
    print(f"   Original equation: y = 3.0000x² - 4.0000x + 5.0000")
    
    # Plot results
    print("\n4. Generating visualization...")
    plot_results(x, y, coefficients)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
