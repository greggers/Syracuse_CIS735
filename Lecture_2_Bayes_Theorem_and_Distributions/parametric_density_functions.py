import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set up the figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('Parametric Density Functions', fontsize=16)

# 1. Gaussian (Normal) Distribution
# ---------------------------------
mu = 5      # mean
sigma = 1   # standard deviation
x_gaussian = np.linspace(0, 10, 1000)
pdf_gaussian = stats.norm.pdf(x_gaussian, mu, sigma)

axes[0].plot(x_gaussian, pdf_gaussian, 'b-', lw=2)
axes[0].fill_between(x_gaussian, pdf_gaussian, alpha=0.2)
axes[0].set_title(f'Gaussian Distribution (μ={mu}, σ={sigma})')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Probability Density')
axes[0].grid(True, alpha=0.3)

# 2. Gamma Distribution
# --------------------
shape = 5    # shape parameter (k or α)
scale = 1    # scale parameter (θ)
x_gamma = np.linspace(0, 15, 1000)
pdf_gamma = stats.gamma.pdf(x_gamma, shape, scale=scale)

axes[1].plot(x_gamma, pdf_gamma, 'r-', lw=2)
axes[1].fill_between(x_gamma, pdf_gamma, alpha=0.2, color='red')
axes[1].set_title(f'Gamma Distribution (k={shape}, θ={scale})')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Probability Density')
axes[1].grid(True, alpha=0.3)

# 3. Binomial Distribution
# -----------------------
n = 20       # number of trials
p = 0.3      # probability of success
x_binomial = np.arange(0, n+1)
pmf_binomial = stats.binom.pmf(x_binomial, n, p)

axes[2].bar(x_binomial, pmf_binomial, alpha=0.7, color='green')
axes[2].set_title(f'Binomial Distribution (n={n}, p={p})')
axes[2].set_xlabel('Number of Successes')
axes[2].set_ylabel('Probability Mass')
axes[2].grid(True, alpha=0.3, axis='y')

# Adjust layout and display
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
plt.savefig('parametric_density_functions.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional information about each distribution
print("Gaussian Distribution:")
print(f"  Mean: {mu}")
print(f"  Variance: {sigma**2}")
print(f"  Support: (-∞, ∞)")
print("\nGamma Distribution:")
print(f"  Mean: {shape * scale}")
print(f"  Variance: {shape * scale**2}")
print(f"  Support: (0, ∞)")
print("\nBinomial Distribution:")
print(f"  Mean: {n * p}")
print(f"  Variance: {n * p * (1-p)}")
print(f"  Support: {{0, 1, 2, ..., {n}}}")
