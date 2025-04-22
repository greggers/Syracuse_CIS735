
# Distance Functions Visualization

This code provides visualizations and implementations of various distance metrics commonly used in machine learning and data analysis:

## 1. Minkowski Distance (Lp Norm)
The Minkowski distance is a generalized metric that includes several common distance measures as special cases based on the parameter p. It's widely used in clustering, classification, and regression tasks.
   * Manhattan Distance (L1): Measures distance as the sum of absolute differences along each dimension, useful in grid-based environments or when diagonal movement isn't allowed.
   * Euclidean Distance (L2): The "straight-line" distance between points in Euclidean space, commonly used in k-means clustering and nearest neighbor algorithms.
   * L3 Norm: A less common variant that gives more weight to larger differences than L2 but less than L1.
   * Chebyshev Distance (L∞): Measures the maximum difference along any coordinate dimension, useful in chess (king's moves) and warehouse logistics.


## 2. Mahalanobis Distance
Measures the distance between a point and a distribution by taking into account the correlations of the dataset. Particularly useful for detecting outliers in multivariate data and pattern recognition.

## 3. Cosine Similarity
Measures the cosine of the angle between two vectors, indicating their directional similarity regardless of magnitude. Commonly used in text analysis, recommendation systems, and high-dimensional data comparison.

## 4. Jaccard Coefficient
Compares the similarity between two sets by calculating the ratio of their intersection to their union. Often used in binary data analysis, document similarity, and comparing sample sets.

## 5. Hausdorff Distance
Measures how far two subsets of a metric space are from each other by finding the greatest of all distances from a point in one set to the closest point in the other set. Useful in shape matching and image comparison.

## Usage
The code generates visualizations for each distance metric and saves them as PNG files. Run the main function to generate all visualizations:

```python
python distance_functions.py
```

This will create five visualization files:
- minkowski_distances.png
- mahalanobis_distance.png
- cosine_similarity.png
- jaccard_similarity.png
- hausdorff_distance.png

## Requirements
* NumPy
* Matplotlib
* SciPy
* Scikit-learn
