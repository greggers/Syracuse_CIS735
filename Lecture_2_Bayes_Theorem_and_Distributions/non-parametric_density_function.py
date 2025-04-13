import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity

# Set random seed for reproducibility
np.random.seed(42)

# Generate data from three Gaussian distributions
# N(1, 0.5) with n=10
data1 = np.random.normal(loc=1, scale=0.5, size=10)

# N(5, 1.5) with n=100
data2 = np.random.normal(loc=5, scale=1.5, size=100)

# N(7, 2.0) with n=50
data3 = np.random.normal(loc=7, scale=2.0, size=50)

# Combine all data
all_data = np.concatenate([data1, data2, data3])

# Define the range for plotting
x_range = np.linspace(-2, 12, 1000)

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle('Non-Parametric Density Estimation', fontsize=16)

# 1. Simple Histogram
# ------------------
axes[0].hist(all_data, bins=20, density=True, alpha=0.7, color='skyblue', 
             edgecolor='black', label='Histogram')

# Plot the true density functions for reference
axes[0].plot(x_range, 
             (10/len(all_data))*stats.norm.pdf(x_range, 1, 0.5), 
             'r-', lw=2, label='N(1, 0.5), n=10')
axes[0].plot(x_range, 
             (100/len(all_data))*stats.norm.pdf(x_range, 5, 1.5), 
             'g-', lw=2, label='N(5, 1.5), n=100')
axes[0].plot(x_range, 
             (50/len(all_data))*stats.norm.pdf(x_range, 7, 2.0), 
             'b-', lw=2, label='N(7, 2.0), n=50')

axes[0].set_title('Simple Histogram')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Average Shifted Histogram (ASH)
# ---------------------------------
def ash_density(data, m, binwidth, x_eval):
    """Compute Average Shifted Histogram density estimate"""
    n = len(data)
    bin_counts = np.zeros((m, len(x_eval)))
    
    for i in range(m):
        shift = i * (binwidth / m)
        bins = np.arange(-2 + shift, 12 + binwidth, binwidth)
        hist, _ = np.histogram(data, bins=bins, density=False)
        
        # Map histogram values to evaluation points
        for j, x in enumerate(x_eval):
            bin_idx = int((x - (-2 + shift)) / binwidth)
            if 0 <= bin_idx < len(hist):
                bin_counts[i, j] = hist[bin_idx]
    
    # Average the shifted histograms
    ash_values = np.mean(bin_counts, axis=0) / (n * binwidth)
    return ash_values

# Compute ASH with different numbers of shifts
m_shifts = 10  # Number of shifts
binwidth = 0.5  # Bin width
ash_density_values = ash_density(all_data, m_shifts, binwidth, x_range)

axes[1].hist(all_data, bins=20, density=True, alpha=0.3, color='lightgray', 
             edgecolor='gray', label='Histogram')
axes[1].plot(x_range, ash_density_values, 'purple', lw=2, 
             label=f'ASH (m={m_shifts}, binwidth={binwidth})')

# Plot the true density functions for reference
axes[1].plot(x_range, 
             (10/len(all_data))*stats.norm.pdf(x_range, 1, 0.5), 
             'r-', lw=1, alpha=0.7, label='N(1, 0.5), n=10')
axes[1].plot(x_range, 
             (100/len(all_data))*stats.norm.pdf(x_range, 5, 1.5), 
             'g-', lw=1, alpha=0.7, label='N(5, 1.5), n=100')
axes[1].plot(x_range, 
             (50/len(all_data))*stats.norm.pdf(x_range, 7, 2.0), 
             'b-', lw=1, alpha=0.7, label='N(7, 2.0), n=50')

axes[1].set_title('Average Shifted Histogram (ASH)')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Kernel Density Estimation with Gaussian Kernel
# ------------------------------------------------
# Different bandwidth values to demonstrate the effect
bandwidths = [0.2, 0.5, 1.0]
colors = ['red', 'green', 'blue']

axes[2].hist(all_data, bins=20, density=True, alpha=0.3, color='lightgray', 
             edgecolor='gray', label='Histogram')

for bw, color in zip(bandwidths, colors):
    # Fit KDE model
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(all_data[:, np.newaxis])
    
    # Score samples returns log-likelihood
    log_dens = kde.score_samples(x_range[:, np.newaxis])
    axes[2].plot(x_range, np.exp(log_dens), color=color, lw=2, 
                 label=f'KDE (bandwidth={bw})')

# Plot the true density functions for reference
axes[2].plot(x_range, 
             (10/len(all_data))*stats.norm.pdf(x_range, 1, 0.5), 
             'r--', lw=1, alpha=0.5, label='N(1, 0.5), n=10')
axes[2].plot(x_range, 
             (100/len(all_data))*stats.norm.pdf(x_range, 5, 1.5), 
             'g--', lw=1, alpha=0.5, label='N(5, 1.5), n=100')
axes[2].plot(x_range, 
             (50/len(all_data))*stats.norm.pdf(x_range, 7, 2.0), 
             'b--', lw=1, alpha=0.5, label='N(7, 2.0), n=50')

axes[2].set_title('Kernel Density Estimation with Gaussian Kernel')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Density')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Adjust layout and display
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
plt.savefig('non_parametric_density_functions.png', dpi=300, bbox_inches='tight')
plt.show()

# Print information about the data and methods
print("Non-Parametric Density Estimation")
print("=================================")
print(f"Data 1: N(1, 0.5), n=10, {len(data1)} samples")
print(f"Data 2: N(5, 1.5), n=100, {len(data2)} samples")
print(f"Data 3: N(7, 2.0), n=50, {len(data3)} samples")
print(f"Total samples: {len(all_data)}")
print("\nMethods demonstrated:")
print("1. Simple Histogram: Basic non-parametric density estimation")
print(f"2. Average Shifted Histogram (ASH): Using {m_shifts} shifts with bin width {binwidth}")
print("3. Kernel Density Estimation: Using Gaussian kernel with different bandwidths")
print("   - Small bandwidth: More detail but potentially overfitting")
print("   - Large bandwidth: Smoother but potentially underfitting")