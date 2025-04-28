import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm, multivariate_normal

# Set random seed for reproducibility
np.random.seed(42)

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))

# 1. 1D Gaussian distribution
# Generate samples from a 1D Gaussian distribution
mean_1d = 0.5
std_1d = 0.15
samples_1d = np.random.normal(mean_1d, std_1d, 10000)

# Filter samples to keep only those in [0, 1] range
samples_1d = samples_1d[(samples_1d >= 0) & (samples_1d <= 1)]

# Plot 1D distribution
ax1 = fig.add_subplot(2, 2, 1)
ax1.hist(samples_1d, bins=50, density=True, alpha=0.7, color='blue')

# Plot the PDF
x = np.linspace(0, 1, 1000)
pdf_1d = norm.pdf(x, mean_1d, std_1d)
ax1.plot(x, pdf_1d, 'r-', lw=2)
ax1.set_title('1D Gaussian Distribution (μ=0.5, σ=0.15)')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.grid(True, alpha=0.3)

# Calculate percentage of points in [0.4, 0.6] range (±0.1 around mean)
pct_1d = np.mean((samples_1d >= 0.4) & (samples_1d <= 0.6)) * 100
ax1.axvspan(0.4, 0.6, alpha=0.2, color='green')
ax1.text(0.1, ax1.get_ylim()[1]*0.9, f"{pct_1d:.1f}% of points\nin [0.4, 0.6]", fontsize=10)

# 2. 2D Gaussian distribution
ax2 = fig.add_subplot(2, 2, 2)

# Generate samples from a 2D Gaussian distribution
mean_2d = [0.5, 0.5]
cov_2d = [[0.15**2, 0], [0, 0.15**2]]  # Diagonal covariance matrix
samples_2d = np.random.multivariate_normal(mean_2d, cov_2d, 10000)

# Filter samples to keep only those in [0,1]x[0,1] range
mask_2d = (samples_2d[:, 0] >= 0) & (samples_2d[:, 0] <= 1) & (samples_2d[:, 1] >= 0) & (samples_2d[:, 1] <= 1)
samples_2d = samples_2d[mask_2d]

# Scatter plot of 2D samples
ax2.scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.5, color='blue')
ax2.set_title('2D Gaussian Distribution (μ=[0.5,0.5], σ=0.15)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# Calculate percentage of points in [0.4, 0.6]x[0.4, 0.6] range
pct_2d = np.mean((samples_2d[:, 0] >= 0.4) & (samples_2d[:, 0] <= 0.6) & 
                 (samples_2d[:, 1] >= 0.4) & (samples_2d[:, 1] <= 0.6)) * 100
ax2.add_patch(plt.Rectangle((0.4, 0.4), 0.2, 0.2, fill=True, alpha=0.2, color='green'))
ax2.text(0.05, 0.95, f"{pct_2d:.1f}% of points\nin [0.4, 0.6]×[0.4, 0.6]", fontsize=10, transform=ax2.transAxes)

# 3. 2D Gaussian density plot
ax3 = fig.add_subplot(2, 2, 3)
x, y = np.mgrid[0:1:100j, 0:1:100j]
pos = np.dstack((x, y))
rv = multivariate_normal(mean_2d, cov_2d)
density_2d = rv.pdf(pos)

# Plot the 2D density
im = ax3.pcolormesh(x, y, density_2d, cmap='viridis', norm=LogNorm())
ax3.set_title('2D Gaussian Density')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
plt.colorbar(im, ax=ax3, label='Density')
ax3.add_patch(plt.Rectangle((0.4, 0.4), 0.2, 0.2, fill=False, edgecolor='red', linewidth=2))

# 4. Curse of dimensionality visualization
ax4 = fig.add_subplot(2, 2, 4)

# Calculate percentage of points in hypercube of side 0.2 centered at 0.5 for different dimensions
dimensions = np.arange(1, 11)
percentages = []

for d in dimensions:
    # Generate samples from d-dimensional Gaussian
    mean_d = np.ones(d) * 0.5
    cov_d = np.eye(d) * (0.15**2)
    samples_d = np.random.multivariate_normal(mean_d, cov_d, 100000)
    
    # Filter samples to keep only those in [0,1]^d hypercube
    mask_d = np.all((samples_d >= 0) & (samples_d <= 1), axis=1)
    samples_d = samples_d[mask_d]
    
    # Calculate percentage of points in [0.4, 0.6]^d hypercube
    mask_center = np.all((samples_d >= 0.4) & (samples_d <= 0.6), axis=1)
    pct_d = np.mean(mask_center) * 100
    percentages.append(pct_d)

# Plot the curse of dimensionality
ax4.plot(dimensions, percentages, 'o-', linewidth=2)
ax4.set_title('Curse of Dimensionality')
ax4.set_xlabel('Dimension')
ax4.set_ylabel('% of Points in Hypercube of Side 0.2')
ax4.set_xticks(dimensions)
ax4.grid(True, alpha=0.3)

# Add text explaining the curse of dimensionality
ax4.text(5, percentages[0]/2, 
         "As dimension increases,\nthe percentage of points\nin the central region\ndecreases exponentially",
         fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.suptitle('Illustration of the Curse of Dimensionality', fontsize=16, y=1.02)
plt.savefig('curse_of_dimensionality.png', dpi=300, bbox_inches='tight')
plt.show()
