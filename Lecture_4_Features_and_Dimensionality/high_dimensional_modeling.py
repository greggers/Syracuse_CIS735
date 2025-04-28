import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.stats import norm, multivariate_normal

# Set random seed for reproducibility
np.random.seed(42)

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))

# 1. Sample size requirements visualization
ax1 = fig.add_subplot(2, 2, 1)

# Data from the table
dimensions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
samples_required = np.array([4, 19, 67, 223, 768, 2790, 10700, 43700, 187000, 842000])

# Plot the relationship between dimensions and required samples
ax1.semilogy(dimensions, samples_required, 'o-', linewidth=2, markersize=8)
ax1.set_title('Required Sample Size vs. Dimensionality')
ax1.set_xlabel('Dimensions')
ax1.set_ylabel('Required Sample Size (log scale)')
ax1.set_xticks(dimensions)
ax1.grid(True, alpha=0.3)

# Add text explaining the relationship
ax1.text(5, 1000, 
         "Sample size requirements\ngrow exponentially\nwith dimensionality",
         fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7))

# 2. Density visualization in different dimensions
ax2 = fig.add_subplot(2, 2, 2)

# Generate fixed number of samples for different dimensions
fixed_samples = 1000
dimensions_to_test = np.arange(1, 11)
density_metric = []

for d in dimensions_to_test:
    # Generate samples in d-dimensional unit hypercube
    samples = np.random.uniform(0, 1, size=(fixed_samples, d))
    
    # Calculate average distance to nearest neighbor as density metric
    distances = []
    for i in range(min(100, fixed_samples)):  # Use subset for efficiency
        dists = np.sqrt(np.sum((samples - samples[i])**2, axis=1))
        dists = dists[dists > 0]  # Remove self-distance
        distances.append(np.min(dists))
    
    density_metric.append(np.mean(distances))

# Plot how density (measured by avg nearest neighbor distance) changes with dimension
ax2.plot(dimensions_to_test, density_metric, 'o-', linewidth=2)
ax2.set_title(f'Average Distance to Nearest Neighbor\n(Fixed {fixed_samples} Samples)')
ax2.set_xlabel('Dimensions')
ax2.set_ylabel('Average Distance')
ax2.set_xticks(dimensions_to_test)
ax2.grid(True, alpha=0.3)

# 3. Model performance visualization
ax3 = fig.add_subplot(2, 2, 3)

# Simulate model performance degradation with dimensionality
# (This is a simplified model - real performance depends on many factors)
dimensions_range = np.arange(1, 11)
performance_metrics = []

base_samples = 100
for d in dimensions_range:
    # Calculate required samples for good performance in this dimension
    required_samples = 4 * (3**d)
    
    # Calculate performance metric (simplified model)
    # Performance degrades when we have fewer samples than required
    performance = min(1.0, base_samples / required_samples)
    performance_metrics.append(performance)

# Plot performance degradation
ax3.plot(dimensions_range, performance_metrics, 'o-', linewidth=2, color='red')
ax3.set_title('Model Performance vs. Dimensionality\n(Fixed Training Set Size)')
ax3.set_xlabel('Dimensions')
ax3.set_ylabel('Relative Performance')
ax3.set_xticks(dimensions_range)
ax3.set_ylim(0, 1.05)
ax3.grid(True, alpha=0.3)

# Add annotation
ax3.text(5, 0.5, 
         "Performance degrades as\ndimensionality increases\nwith fixed sample size",
         fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7))

# 4. Table visualization
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')  # Turn off axis

# Create table data
table_data = {
    'Dimensions': dimensions,
    'Required Samples': samples_required
}
df = pd.DataFrame(table_data)

# Create table
table = ax4.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    bbox=[0.1, 0.1, 0.8, 0.8]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Add title above the table
ax4.set_title('Required Sample Size by Dimension', pad=20)

plt.tight_layout()
plt.suptitle('Modeling in High Dimensional Spaces', fontsize=16, y=1.02)
plt.savefig('high_dimensional_modeling.png', dpi=300, bbox_inches='tight')
plt.show()