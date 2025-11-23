# Machine Learning Portfolio - Python Projects

A collection of machine learning projects demonstrating various algorithms and techniques including regression, classification, clustering, and computer vision using Python and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Projects](#projects)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Project Details](#project-details)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This repository showcases practical implementations of fundamental machine learning algorithms. Each project is designed to demonstrate:

- **Clean, well-documented code** following Python best practices
- **Real-world applications** of ML algorithms
- **Data visualization** techniques
- **Model evaluation** and performance metrics

**Perfect for:**
- Learning machine learning fundamentals
- Portfolio demonstration
- Code reference for ML projects
- Understanding scikit-learn workflows

---

## 🚀 Projects

### 1. **Polynomial Regression for Non-Linear Data**
Demonstrates fitting polynomial curves to non-linear data using scikit-learn.

**Key Features:**
- Polynomial feature transformation
- Linear regression on polynomial features
- Visualization of best-fit curves
- Coefficient extraction and analysis

**Algorithm:** Polynomial Regression (Degree 2)

---

### 2. **K-Nearest Neighbors (KNN) - Iris Classification**
Classic machine learning project using KNN to classify iris flower species.

**Key Features:**
- Train-test split methodology
- KNN classifier implementation
- Model evaluation with accuracy metrics
- Classification report and confusion matrix
- Detailed prediction analysis

**Dataset:** Iris Flower Dataset (150 samples, 4 features, 3 classes)

---

### 3. **MeanShift Clustering with 3D Visualization**
Unsupervised clustering of iris flowers with interactive 3D visualization.

**Key Features:**
- MeanShift clustering algorithm
- Automatic cluster discovery
- 3D scatter plot visualization using matplotlib
- Cluster centroid identification
- Multi-dimensional data visualization

**Algorithm:** MeanShift Clustering

---

### 4. **Real-Time Face Shape Detection**
Computer vision application for detecting and classifying face shapes in real-time.

**Key Features:**
- Real-time webcam processing
- Facial landmark detection (68 points)
- Face shape classification (6 types)
- Live video annotation
- Screenshot capture capability

**Technologies:** OpenCV, dlib, KNN Classification

**Face Types:** Diamond, Oblong, Oval, Round, Square, Triangle

---

## 💻 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Webcam (for face detection project)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-python-portfolio.git
cd ml-python-portfolio

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Additional Setup for Face Detection

Download the required model file:
```bash
# Create models directory
mkdir models

# Download dlib's facial landmark predictor (extract from .bz2)
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Place in: models/shape_predictor_68_face_landmarks.dat
```

**Note:** The face detection project also requires a trained KNN model (`KNN_Model.sav`). See the [Face Detection Documentation](docs/FACE_DETECTION.md) for training instructions.

---

## 🎮 Usage

### Running Individual Projects

#### 1. Polynomial Regression
```bash
python scripts/polynomial_regression.py
```

**Output:**
- Console output with model coefficients
- Matplotlib visualization of fitted curve

---

#### 2. KNN Iris Classification
```bash
python scripts/knn_iris_classification.py
```

**Output:**
- Classification accuracy
- Confusion matrix
- Detailed classification report
- Sample predictions

---

#### 3. MeanShift Clustering
```bash
python scripts/meanshift_clustering_3d.py
```

**Output:**
- Cluster summary statistics
- Interactive 3D visualization
- Cluster centroid coordinates

---

#### 4. Face Shape Detection
```bash
python scripts/face_shape_detector.py
```

**Controls:**
- **Q** - Quit application
- **S** - Save screenshot

**Output:**
- Real-time video with face shape detection
- Facial landmark visualization
- Saved screenshots (optional)

---

## 🛠️ Technologies

### Core Libraries
| Library | Purpose |
|---------|---------|
| **NumPy** | Numerical computing and array operations |
| **scikit-learn** | Machine learning algorithms and tools |
| **Matplotlib** | Data visualization and plotting |
| **OpenCV (cv2)** | Computer vision and image processing |
| **dlib** | Facial landmark detection |
| **imutils** | Image processing utilities |

### Machine Learning Algorithms
- **Linear Regression** (with polynomial features)
- **K-Nearest Neighbors (KNN)**
- **MeanShift Clustering**
- **Classification** (supervised learning)
- **Clustering** (unsupervised learning)

---

## 📊 Project Details

### 1. Polynomial Regression

**Problem:** Fit a curve to non-linear data  
**Solution:** Transform features to polynomial space, then apply linear regression

**Mathematical Formulation:**
```
Original equation: y = 3x² - 4x + 5
Fitted equation: y = ax² + bx + c
```

**Code Highlights:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
```

---

### 2. KNN Iris Classification

**Problem:** Classify iris flowers into 3 species  
**Features:** Sepal length, sepal width, petal length, petal width  
**Classes:** Setosa, Versicolor, Virginica

**Performance Metrics:**
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-Score

**Code Highlights:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
```

---

### 3. MeanShift Clustering

**Problem:** Discover natural groupings in iris data  
**Algorithm:** MeanShift (density-based clustering)  
**Visualization:** 3D scatter plot with cluster centroids

**Key Parameters:**
- `bandwidth=0.85` - Controls cluster granularity

**Code Highlights:**
```python
from sklearn.cluster import MeanShift

ms = MeanShift(bandwidth=0.85)
ms.fit(data)
labels = ms.labels_
centroids = ms.cluster_centers_
```

---

### 4. Face Shape Detection

**Problem:** Classify face shapes in real-time video  
**Pipeline:**
1. Face detection (dlib frontal face detector)
2. Landmark detection (68 facial points)
3. Feature extraction (jaw line ratios)
4. Classification (trained KNN model)

**Feature Engineering:**
- Extract jaw line points (points 2-9)
- Calculate distance ratios
- Normalize features for classification

**Code Highlights:**
```python
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('model.dat')
faces = face_detector(gray_image)
landmarks = landmark_predictor(gray_image, faces[0])
```

---

## 📈 Skills Demonstrated

### Technical Skills
✅ Machine learning algorithm implementation  
✅ Data preprocessing and feature engineering  
✅ Model training and evaluation  
✅ Data visualization (2D and 3D)  
✅ Computer vision and real-time processing  
✅ Python programming best practices  

### Libraries & Tools
✅ scikit-learn (sklearn)  
✅ NumPy for numerical computing  
✅ Matplotlib for visualization  
✅ OpenCV for computer vision  
✅ dlib for facial landmark detection  

### Concepts
✅ Supervised learning (classification, regression)  
✅ Unsupervised learning (clustering)  
✅ Train-test split methodology  
✅ Model evaluation metrics  
✅ Feature transformation  
✅ Real-time data processing  

---

## 📚 Learning Resources

To understand these projects better, refer to:

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [dlib Documentation](http://dlib.net/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest enhancements
- Submit pull requests
- Share your implementations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Ashen Perera**
- GitHub: [@ashensha90](https://github.com/ashensha90)
- LinkedIn: [Ashen Perera](https://www.linkedin.com/in/ashenshanaka/)

---

## 🙏 Acknowledgments

- **scikit-learn team** for the excellent ML library
- **OpenCV community** for computer vision tools
- **UCI Machine Learning Repository** for the Iris dataset
- **Davis King** for the dlib library

---

## 📊 Repository Stats

- **Programming Language:** Python
- **Projects:** 4
- **ML Algorithms:** 4 (Polynomial Regression, KNN, MeanShift, Classification)
- **Lines of Code:** ~800+
- **Documentation:** Comprehensive

---

**⭐ If you find this repository helpful, please consider giving it a star!**

---

*Last Updated: November 2024*
