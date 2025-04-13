import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# Set random seed 
randomSeed = 42

# Set random seed for reproducibility
np.random.seed(randomSeed)

# Generate data from three Gaussian distributions
# N(1, 0.5) with n=10
data1 = np.random.normal(loc=1, scale=0.5, size=10)

# N(5, 1.5) with n=100
data2 = np.random.normal(loc=5, scale=1.5, size=100)

# N(7, 2.0) with n=50
data3 = np.random.normal(loc=7, scale=2.0, size=50)

# Combine all data
all_data = np.concatenate([data1, data2, data3])
all_data = all_data.reshape(-1, 1)  # Reshape for sklearn

# Define the range for plotting
x_range = np.linspace(-2, 12, 1000).reshape(-1, 1)

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle('Semi-Parametric Density Estimation: Gaussian Mixture Models', fontsize=16)

# Function to plot GMM components
def plot_gmm_components(ax, gmm, x, color='red', alpha=0.5):
    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]
        var = gmm.covariances_[i, 0, 0]
        weight = gmm.weights_[i]
        
        # Plot the weighted Gaussian component
        component = weight * stats.norm.pdf(x, mean, np.sqrt(var))
        ax.plot(x, component, '--', color=color, alpha=alpha, 
                label=f'Component {i+1}: μ={mean:.2f}, σ²={var:.2f}, w={weight:.2f}')

# 1. GMM with K=1 (Single Gaussian)
# ---------------------------------
gmm1 = GaussianMixture(n_components=1, random_state=randomSeed)
gmm1.fit(all_data)

# Predict density
log_dens1 = gmm1.score_samples(x_range)
dens1 = np.exp(log_dens1)

axes[0].hist(all_data, bins=20, density=True, alpha=0.7, color='lightgray', 
             edgecolor='gray', label='Data Histogram')
axes[0].plot(x_range, dens1, 'r-', lw=2, label='GMM (K=1)')

# Plot the individual components
plot_gmm_components(axes[0], gmm1, x_range.flatten(), color='darkred', alpha=0.7)

# Plot the true density functions for reference
axes[0].plot(x_range, 
             (10/len(all_data))*stats.norm.pdf(x_range.flatten(), 1, 0.5), 
             'g--', lw=1, alpha=0.5, label='True N(1, 0.5), n=10')
axes[0].plot(x_range, 
             (100/len(all_data))*stats.norm.pdf(x_range.flatten(), 5, 1.5), 
             'b--', lw=1, alpha=0.5, label='True N(5, 1.5), n=100')
axes[0].plot(x_range, 
             (50/len(all_data))*stats.norm.pdf(x_range.flatten(), 7, 2.0), 
             'm--', lw=1, alpha=0.5, label='True N(7, 2.0), n=50')

axes[0].set_title('Gaussian Mixture Model with K=1 Component')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(True, alpha=0.3)

# 2. GMM with K=2 (Two Gaussians)
# -------------------------------
gmm2 = GaussianMixture(n_components=2, random_state=randomSeed)
gmm2.fit(all_data)

# Predict density
log_dens2 = gmm2.score_samples(x_range)
dens2 = np.exp(log_dens2)

axes[1].hist(all_data, bins=20, density=True, alpha=0.7, color='lightgray', 
             edgecolor='gray', label='Data Histogram')
axes[1].plot(x_range, dens2, 'r-', lw=2, label='GMM (K=2)')

# Plot the individual components
plot_gmm_components(axes[1], gmm2, x_range.flatten(), color='darkred', alpha=0.7)

# Plot the true density functions for reference
axes[1].plot(x_range, 
             (10/len(all_data))*stats.norm.pdf(x_range.flatten(), 1, 0.5), 
             'g--', lw=1, alpha=0.5, label='True N(1, 0.5), n=10')
axes[1].plot(x_range, 
             (100/len(all_data))*stats.norm.pdf(x_range.flatten(), 5, 1.5), 
             'b--', lw=1, alpha=0.5, label='True N(5, 1.5), n=100')
axes[1].plot(x_range, 
             (50/len(all_data))*stats.norm.pdf(x_range.flatten(), 7, 2.0), 
             'm--', lw=1, alpha=0.5, label='True N(7, 2.0), n=50')

axes[1].set_title('Gaussian Mixture Model with K=2 Components')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(True, alpha=0.3)

# 3. GMM with K=3 (Three Gaussians)
# --------------------------------
gmm3 = GaussianMixture(n_components=3, random_state=randomSeed)
gmm3.fit(all_data)

# Predict density
log_dens3 = gmm3.score_samples(x_range)
dens3 = np.exp(log_dens3)

axes[2].hist(all_data, bins=20, density=True, alpha=0.7, color='lightgray', 
             edgecolor='gray', label='Data Histogram')
axes[2].plot(x_range, dens3, 'r-', lw=2, label='GMM (K=3)')

# Plot the individual components
plot_gmm_components(axes[2], gmm3, x_range.flatten(), color='darkred', alpha=0.7)

# Plot the true density functions for reference
axes[2].plot(x_range, 
             (10/len(all_data))*stats.norm.pdf(x_range.flatten(), 1, 0.5), 
             'g--', lw=1, alpha=0.5, label='True N(1, 0.5), n=10')
axes[2].plot(x_range, 
             (100/len(all_data))*stats.norm.pdf(x_range.flatten(), 5, 1.5), 
             'b--', lw=1, alpha=0.5, label='True N(5, 1.5), n=100')
axes[2].plot(x_range, 
             (50/len(all_data))*stats.norm.pdf(x_range.flatten(), 7, 2.0), 
             'm--', lw=1, alpha=0.5, label='True N(7, 2.0), n=50')

axes[2].set_title('Gaussian Mixture Model with K=3 Components')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Density')
axes[2].legend(loc='upper right', fontsize=8)
axes[2].grid(True, alpha=0.3)

# Adjust layout and display
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
plt.savefig('semi_parametric_density_functions.png', dpi=300, bbox_inches='tight')
plt.show()

# Print information about the models
print("Semi-Parametric Density Estimation: Gaussian Mixture Models")
print("==========================================================")
print(f"Data 1: N(1, 0.5), n=10, {len(data1)} samples")
print(f"Data 2: N(5, 1.5), n=100, {len(data2)} samples")
print(f"Data 3: N(7, 2.0), n=50, {len(data3)} samples")
print(f"Total samples: {len(all_data)}")

print("\nGMM with K=1:")
print(f"  Mean: {gmm1.means_.flatten()}")
print(f"  Variance: {gmm1.covariances_.flatten()}")
print(f"  Weight: {gmm1.weights_}")
print(f"  BIC: {gmm1.bic(all_data)}")

print("\nGMM with K=2:")
print(f"  Means: {gmm2.means_.flatten()}")
print(f"  Variances: {gmm2.covariances_.flatten()}")
print(f"  Weights: {gmm2.weights_}")
print(f"  BIC: {gmm2.bic(all_data)}")

print("\nGMM with K=3:")
print(f"  Means: {gmm3.means_.flatten()}")
print(f"  Variances: {gmm3.covariances_.flatten()}")
print(f"  Weights: {gmm3.weights_}")
print(f"  BIC: {gmm3.bic(all_data)}")

# Calculate AIC and BIC for model comparison
models = [gmm1, gmm2, gmm3]
n_components = [1, 2, 3]
bic = [m.bic(all_data) for m in models]
aic = [m.aic(all_data) for m in models]

print("\nModel Selection Criteria:")
print("  K\tBIC\tAIC")
for k, b, a in zip(n_components, bic, aic):
    print(f"  {k}\t{b:.2f}\t{a:.2f}")

best_bic = np.argmin(bic) + 1
best_aic = np.argmin(aic) + 1
print(f"\nBest model according to BIC: K={best_bic}")
print(f"Best model according to AIC: K={best_aic}")