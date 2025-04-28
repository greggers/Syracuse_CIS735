import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_blobs, make_classification
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import gaussian_kde

# Set random seed for reproducibility
np.random.seed(42)

def plot_ellipse(ax, mean, cov, color, alpha=0.3, label=None):
    """Plot an ellipse representing a 2D Gaussian distribution."""
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Get the index of the largest eigenvalue
    largest_eigval_idx = np.argmax(eigenvals)
    largest_eigvec = eigenvecs[:, largest_eigval_idx]
    
    # Get the angle of the largest eigenvector
    angle = np.arctan2(largest_eigvec[1], largest_eigvec[0])
    angle = np.degrees(angle)
    
    # Compute width and height of the ellipse (2 standard deviations)
    width, height = 4 * np.sqrt(eigenvals)
    
    # Create the ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor=color, facecolor=color, alpha=alpha, label=label)
    
    ax.add_patch(ellipse)
    
    # Plot the direction of the largest eigenvector
    scale_factor = 2 * np.sqrt(eigenvals[largest_eigval_idx])
    ax.arrow(mean[0], mean[1], 
             largest_eigvec[0] * scale_factor, largest_eigvec[1] * scale_factor,
             head_width=0.2, head_length=0.2, fc=color, ec=color, alpha=0.8)

def plot_lda_direction(ax, X, y, lda, color='purple', label='LDA Direction'):
    """Plot the LDA direction that maximizes class separation."""
    # Get the mean of the data
    mean = np.mean(X, axis=0)
    
    # Get the LDA direction (first component)
    direction = lda.coef_[0]
    
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Scale the direction for visualization
    scale_factor = 5
    
    # Plot the LDA direction
    ax.arrow(mean[0], mean[1], 
             direction[0] * scale_factor, direction[1] * scale_factor,
             head_width=0.3, head_length=0.3, fc=color, ec=color, 
             label=label, linewidth=2)

def project_data(X, direction):
    """Project data onto a direction vector."""
    # Normalize the direction
    direction = direction / np.linalg.norm(direction)
    
    # Project the data
    return np.dot(X, direction)

# Create figure
plt.figure(figsize=(20, 15))

# Example 1: Basic LDA on well-separated data
# ------------------------------------------
# Generate two well-separated classes
X1, y1 = make_blobs(n_samples=[100, 100], centers=[[0, 0], [4, 4]], 
                   cluster_std=[1.0, 1.0], random_state=42)

# Fit LDA
lda1 = LinearDiscriminantAnalysis(n_components=1)
lda1.fit(X1, y1)
X1_lda = lda1.transform(X1)

# Plot original data with class distributions
ax1 = plt.subplot(2, 3, 1)
colors = ['blue', 'red']
class_names = ['Class 0', 'Class 1']

for i, color in enumerate(colors):
    idx = y1 == i
    ax1.scatter(X1[idx, 0], X1[idx, 1], c=color, label=class_names[i], alpha=0.7)
    
    # Calculate and plot class mean and covariance
    class_mean = np.mean(X1[idx], axis=0)
    class_cov = np.cov(X1[idx, 0], X1[idx, 1])
    plot_ellipse(ax1, class_mean, class_cov, color)
    
    # Mark class mean
    ax1.plot(class_mean[0], class_mean[1], 'o', color=color, markersize=10, 
             markeredgecolor='black')

# Plot LDA direction
plot_lda_direction(ax1, X1, y1, lda1)

ax1.set_title('Two Well-Separated Classes with LDA Direction')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot data projected onto LDA direction
ax2 = plt.subplot(2, 3, 2)
for i, color in enumerate(colors):
    idx = y1 == i
    ax2.scatter(X1_lda[idx], np.zeros_like(X1_lda[idx]) + i*0.1, c=color, 
                label=class_names[i], alpha=0.7)
    
    # Plot kernel density estimate
    kde = gaussian_kde(X1_lda[idx].flatten())
    x_range = np.linspace(X1_lda.min(), X1_lda.max(), 1000)
    ax2.plot(x_range, kde(x_range) + i*0.1, color=color, linewidth=2)

ax2.set_title('Data Projected onto LDA Direction')
ax2.set_xlabel('LDA Component 1')
ax2.set_yticks([])
ax2.grid(True, alpha=0.3)
ax2.legend()

# Example 2: LDA vs PCA on overlapping data
# ----------------------------------------
# Generate overlapping classes with specific covariance structure
X2, y2 = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                            n_informative=2, n_clusters_per_class=1, 
                            class_sep=2.0, random_state=42)

# Fit LDA
lda2 = LinearDiscriminantAnalysis(n_components=1)
lda2.fit(X2, y2)
X2_lda = lda2.transform(X2)

# Fit PCA (for comparison)
from sklearn.decomposition import PCA
pca2 = PCA(n_components=1)
pca2.fit(X2)
X2_pca = pca2.transform(X2)

# Plot original data with LDA and PCA directions
ax3 = plt.subplot(2, 3, 3)
for i, color in enumerate(colors):
    idx = y2 == i
    ax3.scatter(X2[idx, 0], X2[idx, 1], c=color, label=class_names[i], alpha=0.7)
    
    # Calculate and plot class mean and covariance
    class_mean = np.mean(X2[idx], axis=0)
    class_cov = np.cov(X2[idx, 0], X2[idx, 1])
    plot_ellipse(ax3, class_mean, class_cov, color)
    
    # Mark class mean
    ax3.plot(class_mean[0], class_mean[1], 'o', color=color, markersize=10, 
             markeredgecolor='black')

# Plot LDA direction
plot_lda_direction(ax3, X2, y2, lda2, color='purple', label='LDA Direction')

# Plot PCA direction
mean = np.mean(X2, axis=0)
pca_direction = pca2.components_[0]
scale_factor = 5
ax3.arrow(mean[0], mean[1], 
         pca_direction[0] * scale_factor, pca_direction[1] * scale_factor,
         head_width=0.3, head_length=0.3, fc='green', ec='green', 
         label='PCA Direction', linewidth=2)

ax3.set_title('LDA vs PCA Directions')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot data projected onto LDA and PCA directions
ax4 = plt.subplot(2, 3, 4)
# LDA projections
for i, color in enumerate(colors):
    idx = y2 == i
    ax4.scatter(X2_lda[idx], np.zeros_like(X2_lda[idx]) + i*0.1, c=color, 
                label=f'{class_names[i]} (LDA)', alpha=0.7)
    
    # Plot kernel density estimate
    kde = gaussian_kde(X2_lda[idx].flatten())
    x_range = np.linspace(X2_lda.min(), X2_lda.max(), 1000)
    ax4.plot(x_range, kde(x_range) + i*0.1, color=color, linewidth=2)

# PCA projections (offset for clarity)
offset = 0.3
for i, color in enumerate(colors):
    idx = y2 == i
    ax4.scatter(X2_pca[idx], np.zeros_like(X2_pca[idx]) + i*0.1 + offset, c=color, 
                marker='s', label=f'{class_names[i]} (PCA)', alpha=0.7)
    
    # Plot kernel density estimate
    kde = gaussian_kde(X2_pca[idx].flatten())
    x_range = np.linspace(X2_pca.min(), X2_pca.max(), 1000)
    ax4.plot(x_range, kde(x_range) + i*0.1 + offset, color=color, linewidth=2, linestyle='--')

ax4.set_title('Data Projected onto LDA vs PCA Directions')
ax4.set_xlabel('Component 1')
ax4.set_yticks([])
ax4.grid(True, alpha=0.3)
ax4.legend()

# Example 3: Multi-class LDA
# -------------------------
# Generate three classes
X3, y3 = make_blobs(n_samples=[100, 100, 100], 
                   centers=[[-5, -5], [0, 0], [5, 5]], 
                   cluster_std=[1.5, 1.5, 1.5], random_state=42)

# Add some rotation to make it more interesting
rotation_matrix = np.array([[0.8, -0.6], [0.6, 0.8]])
X3 = np.dot(X3, rotation_matrix)

# Fit LDA
lda3 = LinearDiscriminantAnalysis(n_components=2)
lda3.fit(X3, y3)
X3_lda = lda3.transform(X3)

# Plot original data
ax5 = plt.subplot(2, 3, 5)
colors_3class = ['blue', 'red', 'green']
class_names_3class = ['Class 0', 'Class 1', 'Class 2']

for i, color in enumerate(colors_3class):
    idx = y3 == i
    ax5.scatter(X3[idx, 0], X3[idx, 1], c=color, label=class_names_3class[i], alpha=0.7)
    
    # Calculate and plot class mean and covariance
    class_mean = np.mean(X3[idx], axis=0)
    class_cov = np.cov(X3[idx, 0], X3[idx, 1])
    plot_ellipse(ax5, class_mean, class_cov, color)
    
    # Mark class mean
    ax5.plot(class_mean[0], class_mean[1], 'o', color=color, markersize=10, 
             markeredgecolor='black')

# Plot LDA directions (for multi-class, we have multiple directions)
mean = np.mean(X3, axis=0)
for i, color in enumerate(['purple', 'orange']):
    direction = lda3.scalings_[:, i]
    scale_factor = 5
    ax5.arrow(mean[0], mean[1], 
             direction[0] * scale_factor, direction[1] * scale_factor,
             head_width=0.3, head_length=0.3, fc=color, ec=color, 
             label=f'LDA Direction {i+1}', linewidth=2)

ax5.set_title('Multi-class LDA Directions')
ax5.set_xlabel('Feature 1')
ax5.set_ylabel('Feature 2')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot data projected onto LDA space
ax6 = plt.subplot(2, 3, 6)
for i, color in enumerate(colors_3class):
    idx = y3 == i
    ax6.scatter(X3_lda[idx, 0], X3_lda[idx, 1], c=color, 
                label=class_names_3class[i], alpha=0.7)
    
    # Calculate and plot class mean and covariance in LDA space
    class_mean_lda = np.mean(X3_lda[idx], axis=0)
    class_cov_lda = np.cov(X3_lda[idx, 0], X3_lda[idx, 1])
    plot_ellipse(ax6, class_mean_lda, class_cov_lda, color)
    
    # Mark class mean
    ax6.plot(class_mean_lda[0], class_mean_lda[1], 'o', color=color, markersize=10, 
             markeredgecolor='black')

ax6.set_title('Data Projected onto 2D LDA Space')
ax6.set_xlabel('LDA Component 1')
ax6.set_ylabel('LDA Component 2')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Add explanatory text for the first figure
plt.figtext(0.5, 0.01, 
            "Linear Discriminant Analysis (LDA) finds directions that maximize between-class variance\n"
            "while minimizing within-class variance. Unlike PCA, LDA is a supervised technique that\n"
            "uses class labels to find the most discriminative projection.",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.suptitle('Linear Discriminant Analysis (LDA) Visualization', fontsize=16, y=1.02)
plt.savefig('lda_visualization.png', dpi=300, bbox_inches='tight')

# Create a second figure to illustrate Fisher's criterion
plt.figure(figsize=(15, 10))

# Generate two classes with different separability
X_easy, y_easy = make_blobs(n_samples=[100, 100], centers=[[0, 0], [5, 5]], 
                           cluster_std=[1.0, 1.0], random_state=42)

X_hard, y_hard = make_blobs(n_samples=[100, 100], centers=[[0, 0], [3, 3]], 
                           cluster_std=[1.5, 1.5], random_state=42)

# Fit LDA to both datasets
lda_easy = LinearDiscriminantAnalysis(n_components=1)
lda_easy.fit(X_easy, y_easy)
X_easy_lda = lda_easy.transform(X_easy)

lda_hard = LinearDiscriminantAnalysis(n_components=1)
lda_hard.fit(X_hard, y_hard)
X_hard_lda = lda_hard.transform(X_hard)

# Plot the original data and LDA directions for easy separation case
ax1 = plt.subplot(2, 2, 1)
for i, color in enumerate(colors):
    idx = y_easy == i
    ax1.scatter(X_easy[idx, 0], X_easy[idx, 1], c=color, label=class_names[i], alpha=0.7)
    
    # Calculate and plot class mean and covariance
    class_mean = np.mean(X_easy[idx], axis=0)
    class_cov = np.cov(X_easy[idx, 0], X_easy[idx, 1])
    plot_ellipse(ax1, class_mean, class_cov, color)
    
    # Mark class mean
    ax1.plot(class_mean[0], class_mean[1], 'o', color=color, markersize=10, 
             markeredgecolor='black')

# Plot LDA direction
plot_lda_direction(ax1, X_easy, y_easy, lda_easy, color='purple', label='LDA Direction')

ax1.set_title('Easy Separation Case')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot the original data and LDA directions for hard separation case
ax2 = plt.subplot(2, 2, 2)
for i, color in enumerate(colors):
    idx = y_hard == i
    ax2.scatter(X_hard[idx, 0], X_hard[idx, 1], c=color, label=class_names[i], alpha=0.7)
    
    # Calculate and plot class mean and covariance
    class_mean = np.mean(X_hard[idx], axis=0)
    class_cov = np.cov(X_hard[idx, 0], X_hard[idx, 1])
    plot_ellipse(ax2, class_mean, class_cov, color)
    
    # Mark class mean
    ax2.plot(class_mean[0], class_mean[1], 'o', color=color, markersize=10, 
             markeredgecolor='black')

# Plot LDA direction
plot_lda_direction(ax2, X_hard, y_hard, lda_hard, color='purple', label='LDA Direction')

ax2.set_title('Hard Separation Case')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot projections for easy separation case
ax3 = plt.subplot(2, 2, 3)
for i, color in enumerate(colors):
    idx = y_easy == i
    ax3.scatter(X_easy_lda[idx], np.zeros_like(X_easy_lda[idx]) + i*0.1, c=color, 
                label=class_names[i], alpha=0.7)
    
    # Plot kernel density estimate
    kde = gaussian_kde(X_easy_lda[idx].flatten())
    x_range = np.linspace(X_easy_lda.min(), X_easy_lda.max(), 1000)
    ax3.plot(x_range, kde(x_range) + i*0.1, color=color, linewidth=2)

# Calculate and display Fisher's criterion (J)
class0_mean = np.mean(X_easy_lda[y_easy == 0])
class1_mean = np.mean(X_easy_lda[y_easy == 1])
class0_var = np.var(X_easy_lda[y_easy == 0])
class1_var = np.var(X_easy_lda[y_easy == 1])
between_class_var = (class1_mean - class0_mean)**2
within_class_var = class0_var + class1_var
fisher_j_easy = between_class_var / within_class_var

ax3.set_title(f'Easy Separation Projection (J = {fisher_j_easy:.2f})')
ax3.set_xlabel('LDA Component 1')
ax3.set_yticks([])
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot projections for hard separation case
ax4 = plt.subplot(2, 2, 4)
for i, color in enumerate(colors):
    idx = y_hard == i
    ax4.scatter(X_hard_lda[idx], np.zeros_like(X_hard_lda[idx]) + i*0.1, c=color, 
                label=class_names[i], alpha=0.7)
    
    # Plot kernel density estimate
    kde = gaussian_kde(X_hard_lda[idx].flatten())
    x_range = np.linspace(X_hard_lda.min(), X_hard_lda.max(), 1000)
    ax4.plot(x_range, kde(x_range) + i*0.1, color=color, linewidth=2)

# Calculate and display Fisher's criterion (J)
class0_mean = np.mean(X_hard_lda[y_hard == 0])
class1_mean = np.mean(X_hard_lda[y_hard == 1])
class0_var = np.var(X_hard_lda[y_hard == 0])
class1_var = np.var(X_hard_lda[y_hard == 1])
between_class_var = (class1_mean - class0_mean)**2
within_class_var = class0_var + class1_var
fisher_j_hard = between_class_var / within_class_var

ax4.set_title(f'Hard Separation Projection (J = {fisher_j_hard:.2f})')
ax4.set_xlabel('LDA Component 1')
ax4.set_yticks([])
ax4.grid(True, alpha=0.3)
ax4.legend()

# Add explanatory text
plt.figtext(0.5, 0.01, 
            "Fisher's criterion (J) measures the quality of class separation:\n"
            "J = (between-class variance) / (within-class variance)\n"
            "Higher J values indicate better class separation. LDA finds the direction that maximizes J.",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.suptitle("Fisher's Criterion in LDA", fontsize=16, y=1.02)
plt.savefig('fisher_criterion.png', dpi=300, bbox_inches='tight')

plt.show()

