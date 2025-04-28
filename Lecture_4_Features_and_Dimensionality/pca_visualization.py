import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_moons
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# Updated Arrow3D class compatible with newer Matplotlib versions
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

# Create figure
plt.figure(figsize=(20, 15))

# Example 1: Basic PCA on 2D data
# ------------------------------
# Generate 2D data with a clear direction of maximum variance
X1 = np.dot(np.random.randn(200, 2), [[3, 1], [1, 2]]) + np.array([3, 3])

# Compute PCA
pca = PCA(n_components=2)
pca.fit(X1)
X1_pca = pca.transform(X1)

# Project data onto the first principal component
X1_projected = np.outer(X1_pca[:, 0], pca.components_[0, :]) + pca.mean_

# Plot original data and principal components
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(X1[:, 0], X1[:, 1], alpha=0.7, label='Original data')
ax1.arrow(pca.mean_[0], pca.mean_[1], 
          pca.components_[0, 0] * 5, pca.components_[0, 1] * 5, 
          head_width=0.5, head_length=0.5, fc='red', ec='red', 
          label='First PC')
ax1.arrow(pca.mean_[0], pca.mean_[1], 
          pca.components_[1, 0] * 5, pca.components_[1, 1] * 5, 
          head_width=0.5, head_length=0.5, fc='green', ec='green', 
          label='Second PC')

# Draw projection lines from points to the first PC
for i in range(0, len(X1), 10):  # Plot every 10th point for clarity
    ax1.plot([X1[i, 0], X1_projected[i, 0]], [X1[i, 1], X1_projected[i, 1]], 
             'k--', alpha=0.3)

ax1.set_title('2D Data with Principal Components')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot data projected onto first principal component
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(X1_pca[:, 0], np.zeros_like(X1_pca[:, 0]), alpha=0.7)
ax2.set_title('Data Projected onto First Principal Component')
ax2.set_xlabel('First Principal Component')
ax2.set_yticks([])
ax2.grid(True, alpha=0.3)

# Plot variance explained
ax3 = plt.subplot(2, 3, 3)
explained_variance_ratio = pca.explained_variance_ratio_
ax3.bar([1, 2], explained_variance_ratio, alpha=0.7)
ax3.set_title('Explained Variance Ratio')
ax3.set_xlabel('Principal Component')
ax3.set_ylabel('Explained Variance Ratio')
ax3.set_xticks([1, 2])
ax3.grid(True, alpha=0.3)
for i, v in enumerate(explained_variance_ratio):
    ax3.text(i+1, v + 0.01, f'{v:.2f}', ha='center')

# Example 2: PCA on clustered data
# -------------------------------
# Generate clustered data
X2, y2 = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# Compute PCA
pca2 = PCA(n_components=2)
pca2.fit(X2)
X2_pca = pca2.transform(X2)

# Project data onto the first principal component
X2_projected = np.outer(X2_pca[:, 0], pca2.components_[0, :]) + pca2.mean_

# Plot original clustered data
ax4 = plt.subplot(2, 3, 4)
scatter = ax4.scatter(X2[:, 0], X2[:, 1], c=y2, cmap='viridis', alpha=0.7)
ax4.arrow(pca2.mean_[0], pca2.mean_[1], 
          pca2.components_[0, 0] * 5, pca2.components_[0, 1] * 5, 
          head_width=0.5, head_length=0.5, fc='red', ec='red')
ax4.arrow(pca2.mean_[0], pca2.mean_[1], 
          pca2.components_[1, 0] * 5, pca2.components_[1, 1] * 5, 
          head_width=0.5, head_length=0.5, fc='green', ec='green')

# Draw projection lines for a subset of points
for i in range(0, len(X2), 20):
    ax4.plot([X2[i, 0], X2_projected[i, 0]], [X2[i, 1], X2_projected[i, 1]], 
             'k--', alpha=0.3)

ax4.set_title('Clustered Data with Principal Components')
ax4.set_xlabel('Feature 1')
ax4.set_ylabel('Feature 2')
ax4.grid(True, alpha=0.3)
legend1 = ax4.legend(*scatter.legend_elements(), title="Clusters")
ax4.add_artist(legend1)

# Plot data projected onto first principal component (colored by cluster)
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(X2_pca[:, 0], np.zeros_like(X2_pca[:, 0]), c=y2, cmap='viridis', alpha=0.7)
ax5.set_title('Clustered Data Projected onto First PC')
ax5.set_xlabel('First Principal Component')
ax5.set_yticks([])
ax5.grid(True, alpha=0.3)

# Example 3: PCA on 3D data
# ------------------------
# Generate 3D data
X3 = np.dot(np.random.randn(200, 3), [[3, 1, 0.5], [1, 2, 0.3], [0.5, 0.3, 1]]) + np.array([3, 3, 3])

# Compute PCA
pca3 = PCA(n_components=3)
pca3.fit(X3)
X3_pca = pca3.transform(X3)

# Create 3D plot
ax6 = plt.subplot(2, 3, 6, projection='3d')
ax6.scatter(X3[:, 0], X3[:, 1], X3[:, 2], alpha=0.7)

# Plot principal components as arrows
arrow_length = 5
for i, (eigenvector, color) in enumerate(zip(pca3.components_, ['red', 'green', 'blue'])):
    arrow = Arrow3D([pca3.mean_[0], pca3.mean_[0] + eigenvector[0] * arrow_length],
                    [pca3.mean_[1], pca3.mean_[1] + eigenvector[1] * arrow_length],
                    [pca3.mean_[2], pca3.mean_[2] + eigenvector[2] * arrow_length],
                    mutation_scale=20, lw=2, arrowstyle='-|>', color=color)
    ax6.add_artist(arrow)
    
ax6.set_title('3D Data with Principal Components')
ax6.set_xlabel('Feature 1')
ax6.set_ylabel('Feature 2')
ax6.set_zlabel('Feature 3')

# Create a second figure for reconstruction visualization
plt.figure(figsize=(15, 10))

# Generate 2D data with a clear pattern
X4, _ = make_moons(n_samples=500, noise=0.1, random_state=42)

# Compute PCA
pca4 = PCA(n_components=2)
pca4.fit(X4)
X4_pca = pca4.transform(X4)

# Reconstruct using only the first principal component
X4_1d = X4_pca.copy()
X4_1d[:, 1] = 0  # Zero out the second component
X4_reconstructed = pca4.inverse_transform(X4_1d)

# Plot original data
ax7 = plt.subplot(2, 2, 1)
ax7.scatter(X4[:, 0], X4[:, 1], alpha=0.7, label='Original data')
ax7.arrow(pca4.mean_[0], pca4.mean_[1], 
          pca4.components_[0, 0] * 3, pca4.components_[0, 1] * 3, 
          head_width=0.2, head_length=0.2, fc='red', ec='red', 
          label='First PC')
ax7.arrow(pca4.mean_[0], pca4.mean_[1], 
          pca4.components_[1, 0] * 3, pca4.components_[1, 1] * 3, 
          head_width=0.2, head_length=0.2, fc='green', ec='green', 
          label='Second PC')
ax7.set_title('Original Moon-Shaped Data')
ax7.set_xlabel('Feature 1')
ax7.set_ylabel('Feature 2')
ax7.grid(True, alpha=0.3)
ax7.legend()

# Plot data projected onto first principal component (1D)
ax8 = plt.subplot(2, 2, 2)
ax8.scatter(X4_pca[:, 0], np.zeros_like(X4_pca[:, 0]), alpha=0.7)
ax8.set_title('Data Projected onto First Principal Component (1D)')
ax8.set_xlabel('First Principal Component')
ax8.set_yticks([])
ax8.grid(True, alpha=0.3)

# Plot reconstructed data using only the first principal component
ax9 = plt.subplot(2, 2, 3)
ax9.scatter(X4_reconstructed[:, 0], X4_reconstructed[:, 1], alpha=0.7, 
            label='Reconstructed data')
ax9.scatter(X4[:, 0], X4[:, 1], alpha=0.2, color='gray', label='Original data')
ax9.set_title('Data Reconstructed from First PC Only')
ax9.set_xlabel('Feature 1')
ax9.set_ylabel('Feature 2')
ax9.grid(True, alpha=0.3)
ax9.legend()

# Plot information loss visualization
ax10 = plt.subplot(2, 2, 4)
for i in range(50):  # Plot lines for a subset of points
    ax10.plot([X4[i, 0], X4_reconstructed[i, 0]], [X4[i, 1], X4_reconstructed[i, 1]], 
              'k--', alpha=0.3)
ax10.scatter(X4[:, 0], X4[:, 1], alpha=0.7, label='Original data')
ax10.scatter(X4_reconstructed[:, 0], X4_reconstructed[:, 1], alpha=0.7, 
             label='Reconstructed data')
ax10.set_title('Information Loss in Dimensionality Reduction')
ax10.set_xlabel('Feature 1')
ax10.set_ylabel('Feature 2')
ax10.grid(True, alpha=0.3)
ax10.legend()

# Add explanatory text
plt.figtext(0.5, 0.01, 
            "PCA finds directions (principal components) that maximize variance in the data.\n"
            "Dimensionality reduction occurs by projecting data onto fewer principal components.\n"
            "This process preserves as much information as possible, but some information is inevitably lost.",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('pca_reconstruction.png', dpi=300, bbox_inches='tight')

# Adjust layout for the first figure
plt.figure(1)
plt.tight_layout()
plt.suptitle('Principal Component Analysis (PCA) Visualization', fontsize=16, y=1.02)
plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')

plt.show()
