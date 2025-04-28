import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Prior probabilities
P_D = 0.90
P_not_D = 1 - P_D

# Gaussian parameters
mu_D, sigma_D = 10, 1.5
mu_not_D, sigma_not_D = 5, 1.0

# Observed test result
x_obs = 7

# Range of x values for plotting
x = np.linspace(0, 15, 500)

# Prior likelihoods (PDFs)
pdf_D = norm.pdf(x, mu_D, sigma_D)
pdf_not_D = norm.pdf(x, mu_not_D, sigma_not_D)

# Plot 1: Prior probability distributions
plt.figure(figsize=(10, 5))
plt.plot(x, pdf_D, label='P(x|D) (Diseased)', color='red')
plt.plot(x, pdf_not_D, label='P(x|¬D) (Healthy)', color='blue')
plt.axvline(x_obs, color='black', linestyle='--', label=f'x = {x_obs}')
plt.title('Prior Distributions: P(x|D) and P(x|¬D)')
plt.xlabel('Test Result Value (x)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute posterior probabilities across x range
posterior_D = (pdf_D * P_D) / (pdf_D * P_D + pdf_not_D * P_not_D)
posterior_not_D = 1 - posterior_D

# Plot 2: Posterior probability given x
plt.figure(figsize=(10, 5))
plt.plot(x, posterior_D, label='P(D|x)', color='green')
plt.plot(x, posterior_not_D, label='P(¬D|x)', color='purple')
plt.axvline(x_obs, color='black', linestyle='--', label=f'x = {x_obs}')
plt.title('Posterior Probabilities: P(D|x) and P(¬D|x)')
plt.xlabel('Test Result Value (x)')
plt.ylabel('Posterior Probability')
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print posterior for observed value
x_p = x_obs
pdf_x_given_D = norm.pdf(x_p, mu_D, sigma_D)
pdf_x_given_not_D = norm.pdf(x_p, mu_not_D, sigma_not_D)
P_D_given_x = (pdf_x_given_D * P_D) / (
    pdf_x_given_D * P_D + pdf_x_given_not_D * P_not_D
)
print(f"P(Disease | x={x_p}) = {P_D_given_x:.4f}")
