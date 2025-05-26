import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ModelSelectionDemo:
    """
    Demonstration of Akaike's Information Criterion (AIC) and 
    Bayesian Information Criterion (BIC) for model selection.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def calculate_aic(self, n, mse, k):
        """
        Calculate Akaike's Information Criterion (AIC)
        
        AIC = 2k - 2ln(L)
        For linear regression with Gaussian errors:
        AIC = n * ln(MSE) + 2k
        
        Args:
            n: number of data points
            mse: mean squared error
            k: number of parameters (complexity)
        
        Returns:
            AIC value
        """
        return n * np.log(mse) + 2 * k
    
    def calculate_bic(self, n, mse, k):
        """
        Calculate Bayesian Information Criterion (BIC)
        
        BIC = k * ln(n) - 2ln(L)
        For linear regression with Gaussian errors:
        BIC = n * ln(MSE) + k * ln(n)
        
        Args:
            n: number of data points
            mse: mean squared error
            k: number of parameters (complexity)
        
        Returns:
            BIC value
        """
        return n * np.log(mse) + 0.5 * k * np.log(n)
    
    def generate_synthetic_data(self, n_samples=50, noise_level=0.3):
        """
        Generate synthetic data with known underlying function
        True function: y = 0.5*x^3 - 2*x^2 + x + noise
        """
        X = np.linspace(-2, 3, n_samples)
        # True underlying function (cubic)
        y_true = 0.5 * X**3 - 2 * X**2 + X
        # Add noise
        y = y_true + np.random.normal(0, noise_level, n_samples)
        
        return X.reshape(-1, 1), y, y_true
    
    def fit_polynomial_models(self, X, y, max_degree=10):
        """
        Fit polynomial models of different degrees and calculate AIC/BIC
        """
        n_samples = len(y)
        degrees = range(1, max_degree + 1)
        
        results = {
            'degrees': [],
            'mse': [],
            'aic': [],
            'bic': [],
            'n_params': [],
            'models': []
        }
        
        for degree in degrees:
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly_features.fit_transform(X)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Make predictions
            y_pred = model.predict(X_poly)
            
            # Calculate MSE
            mse = mean_squared_error(y, y_pred)
            
            # Number of parameters (coefficients + intercept)
            k = X_poly.shape[1]
            
            # Calculate AIC and BIC
            aic = self.calculate_aic(n_samples, mse, k)
            bic = self.calculate_bic(n_samples, mse, k)
            
            # Store results
            results['degrees'].append(degree)
            results['mse'].append(mse)
            results['aic'].append(aic)
            results['bic'].append(bic)
            results['n_params'].append(k)
            results['models'].append((poly_features, model))
        
        return results
    
    def demonstrate_sample_size_effect(self):
        """
        Demonstrate how AIC and BIC behave differently with varying sample sizes
        """
        sample_sizes = [20, 50, 100, 200, 500]
        max_degree = 8
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AIC vs BIC: Effect of Sample Size on Model Selection', fontsize=16)
        
        for i, n_samples in enumerate(sample_sizes):
            if i >= 5:  # Only plot first 5
                break
                
            # Generate data
            X, y, y_true = self.generate_synthetic_data(n_samples=n_samples)
            
            # Fit models
            results = self.fit_polynomial_models(X, y, max_degree=max_degree)
            
            # Plot results
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Normalize AIC and BIC for comparison
            aic_norm = np.array(results['aic']) - min(results['aic'])
            bic_norm = np.array(results['bic']) - min(results['bic'])
            
            ax.plot(results['degrees'], aic_norm, 'b-o', label='AIC', linewidth=2)
            ax.plot(results['degrees'], bic_norm, 'r-s', label='BIC', linewidth=2)
            
            # Mark optimal models
            aic_optimal = results['degrees'][np.argmin(results['aic'])]
            bic_optimal = results['degrees'][np.argmin(results['bic'])]
            
            ax.axvline(aic_optimal, color='blue', linestyle='--', alpha=0.7, 
                      label=f'AIC optimal: degree {aic_optimal}')
            ax.axvline(bic_optimal, color='red', linestyle='--', alpha=0.7,
                      label=f'BIC optimal: degree {bic_optimal}')
            ax.axvline(3, color='green', linestyle=':', alpha=0.7,
                      label='True degree: 3')
            
            ax.set_title(f'n = {n_samples}')
            ax.set_xlabel('Polynomial Degree')
            ax.set_ylabel('Information Criterion (normalized)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(sample_sizes) == 5:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig('Lecture_8_Method_Selection/aic_bic_sample_size_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate_complexity_penalty(self):
        """
        Demonstrate how AIC and BIC penalize model complexity differently
        """
        # Generate data
        X, y, y_true = self.generate_synthetic_data(n_samples=100)
        
        # Fit models
        results = self.fit_polynomial_models(X, y, max_degree=12)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AIC vs BIC: Model Selection Comparison', fontsize=16)
        
        # Plot 1: AIC and BIC curves
        ax1 = axes[0, 0]
        ax1.plot(results['degrees'], results['aic'], 'b-o', label='AIC', linewidth=2)
        ax1.plot(results['degrees'], results['bic'], 'r-s', label='BIC', linewidth=2)
        
        # Mark optimal models
        aic_optimal = results['degrees'][np.argmin(results['aic'])]
        bic_optimal = results['degrees'][np.argmin(results['bic'])]
        
        ax1.axvline(aic_optimal, color='blue', linestyle='--', alpha=0.7)
        ax1.axvline(bic_optimal, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(3, color='green', linestyle=':', alpha=0.7, label='True degree: 3')
        
        ax1.set_xlabel('Polynomial Degree')
        ax1.set_ylabel('Information Criterion')
        ax1.set_title('AIC vs BIC')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MSE vs Model Complexity
        ax2 = axes[0, 1]
        ax2.plot(results['degrees'], results['mse'], 'g-^', label='MSE', linewidth=2)
        ax2.axvline(aic_optimal, color='blue', linestyle='--', alpha=0.7, label=f'AIC optimal: {aic_optimal}')
        ax2.axvline(bic_optimal, color='red', linestyle='--', alpha=0.7, label=f'BIC optimal: {bic_optimal}')
        ax2.axvline(3, color='green', linestyle=':', alpha=0.7, label='True degree: 3')
        
        ax2.set_xlabel('Polynomial Degree')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('Model Fit Quality')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Penalty terms comparison
        ax3 = axes[1, 0]
        n_samples = len(y)
        aic_penalty = 2 * np.array(results['n_params'])
        bic_penalty = np.array(results['n_params']) * np.log(n_samples)
        
        ax3.plot(results['degrees'], aic_penalty, 'b-o', label='AIC penalty: 2k', linewidth=2)
        ax3.plot(results['degrees'], bic_penalty, 'r-s', label=f'BIC penalty: k*ln({n_samples})', linewidth=2)
        
        ax3.set_xlabel('Polynomial Degree')
        ax3.set_ylabel('Penalty Term')
        ax3.set_title('Complexity Penalties')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model predictions comparison
        ax4 = axes[1, 1]
        X_plot = np.linspace(-2, 3, 200).reshape(-1, 1)
        
        # Plot data
        ax4.scatter(X, y, alpha=0.6, color='gray', label='Data')
        
        # Plot true function
        y_true_plot = 0.5 * X_plot.flatten()**3 - 2 * X_plot.flatten()**2 + X_plot.flatten()
        ax4.plot(X_plot, y_true_plot, 'g-', linewidth=3, label='True function (degree 3)')
        
        # Plot AIC optimal model
        aic_idx = np.argmin(results['aic'])
        poly_features_aic, model_aic = results['models'][aic_idx]
        X_plot_poly_aic = poly_features_aic.transform(X_plot)
        y_pred_aic = model_aic.predict(X_plot_poly_aic)
        ax4.plot(X_plot, y_pred_aic, 'b--', linewidth=2, 
                label=f'AIC optimal (degree {aic_optimal})')
        
        # Plot BIC optimal model
        bic_idx = np.argmin(results['bic'])
        poly_features_bic, model_bic = results['models'][bic_idx]
        X_plot_poly_bic = poly_features_bic.transform(X_plot)
        y_pred_bic = model_bic.predict(X_plot_poly_bic)
        ax4.plot(X_plot, y_pred_bic, 'r:', linewidth=2, 
                label=f'BIC optimal (degree {bic_optimal})')
        
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title('Model Predictions Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Lecture_8_Method_Selection/aic_bic_complexity_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        return results, aic_optimal, bic_optimal
    
    def print_detailed_results(self, results, aic_optimal, bic_optimal):
        """
        Print detailed numerical results
        """
        print("\n" + "="*80)
        print("DETAILED MODEL SELECTION RESULTS")
        print("="*80)
        
        print(f"\nData: n = {len(results['degrees'])} models evaluated")
        print(f"True underlying function: y = 0.5*x³ - 2*x² + x + noise")
        print(f"True polynomial degree: 3")
        
        print(f"\nAIC optimal model: Degree {aic_optimal}")
        print(f"BIC optimal model: Degree {bic_optimal}")
        
        print(f"\nDetailed Results:")
        print(f"{'Degree':<8} {'Params':<8} {'MSE':<12} {'AIC':<12} {'BIC':<12}")
        print("-" * 60)
        
        for i, degree in enumerate(results['degrees']):
            marker_aic = " *" if degree == aic_optimal else ""
            marker_bic = " **" if degree == bic_optimal else ""
            marker = marker_aic + marker_bic
            
            print(f"{degree:<8} {results['n_params'][i]:<8} "
                  f"{results['mse'][i]:<12.6f} {results['aic'][i]:<12.2f} "
                  f"{results['bic'][i]:<12.2f}{marker}")
        
        print("\n* = AIC optimal, ** = BIC optimal")
        
        # Calculate and display the difference in penalties
        n_samples = 100  # From our demo
        print(f"\nPenalty Comparison (for degree {max(aic_optimal, bic_optimal)}):")
        k = max(aic_optimal, bic_optimal) + 1  # +1 for intercept
        aic_penalty = 2 * k
        bic_penalty = k * np.log(n_samples)
        print(f"AIC penalty: 2k = 2 × {k} = {aic_penalty}")
        print(f"BIC penalty: k×ln(n) = {k} × ln({n_samples}) = {bic_penalty:.2f}")
        print(f"BIC penalty is {bic_penalty/aic_penalty:.2f}x larger than AIC penalty")
        
        print(f"\nKey Insights:")
        print(f"• AIC tends to select more complex models (lower penalty)")
        print(f"• BIC tends to select simpler models (higher penalty)")
        print(f"• As sample size increases, BIC penalty grows while AIC penalty stays constant")
        print(f"• BIC is more conservative and often closer to the true model")
    
    def demonstrate_different_scenarios(self):
        """
        Demonstrate scenarios where AIC and BIC give different recommendations
        """
        scenarios = [
            {"n_samples": 30, "noise": 0.2, "title": "Small Sample, Low Noise"},
            {"n_samples": 30, "noise": 0.5, "title": "Small Sample, High Noise"},
            {"n_samples": 200, "noise": 0.2, "title": "Large Sample, Low Noise"},
            {"n_samples": 200, "noise": 0.5, "title": "Large Sample, High Noise"}
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AIC vs BIC in Different Scenarios', fontsize=16)
        
        results_summary = []
        
        for i, scenario in enumerate(scenarios):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Generate data for this scenario
            X, y, y_true = self.generate_synthetic_data(
                n_samples=scenario["n_samples"], 
                noise_level=scenario["noise"]
            )
            
            # Fit models
            results = self.fit_polynomial_models(X, y, max_degree=10)
            
            # Find optimal models
            aic_optimal = results['degrees'][np.argmin(results['aic'])]
            bic_optimal = results['degrees'][np.argmin(results['bic'])]
            
            # Store results
            results_summary.append({
                'scenario': scenario['title'],
                'n_samples': scenario['n_samples'],
                'noise': scenario['noise'],
                'aic_optimal': aic_optimal,
                'bic_optimal': bic_optimal,
                'difference': abs(aic_optimal - bic_optimal)
            })
            
            # Normalize for plotting
            aic_norm = np.array(results['aic']) - min(results['aic'])
            bic_norm = np.array(results['bic']) - min(results['bic'])
            
            # Plot
            ax.plot(results['degrees'], aic_norm, 'b-o', label='AIC', linewidth=2)
            ax.plot(results['degrees'], bic_norm, 'r-s', label='BIC', linewidth=2)
            
            # Mark optimal points
            ax.axvline(aic_optimal, color='blue', linestyle='--', alpha=0.7)
            ax.axvline(bic_optimal, color='red', linestyle='--', alpha=0.7)
            ax.axvline(3, color='green', linestyle=':', alpha=0.7, label='True: 3')
            
            ax.set_title(f"{scenario['title']}\nAIC: {aic_optimal}, BIC: {bic_optimal}")
            ax.set_xlabel('Polynomial Degree')
            ax.set_ylabel('Information Criterion (normalized)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Lecture_8_Method_Selection/aic_bic_scenarios.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n" + "="*80)
        print("SCENARIO COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Scenario':<25} {'n':<5} {'Noise':<7} {'AIC':<5} {'BIC':<5} {'Diff':<6}")
        print("-" * 80)
        
        for result in results_summary:
            print(f"{result['scenario']:<25} {result['n_samples']:<5} "
                  f"{result['noise']:<7} {result['aic_optimal']:<5} "
                  f"{result['bic_optimal']:<5} {result['difference']:<6}")
        
        return results_summary

def main():
    """
    Main demonstration function
    """
    print("="*80)
    print("MODEL SELECTION DEMO: AKAIKE'S INFORMATION CRITERION (AIC) vs")
    print("BAYESIAN INFORMATION CRITERION (BIC)")
    print("="*80)
    
    # Create demo instance
    demo = ModelSelectionDemo(random_state=42)
    
    print("\n1. COMPLEXITY PENALTY DEMONSTRATION")
    print("-" * 50)
    results, aic_optimal, bic_optimal = demo.demonstrate_complexity_penalty()
    demo.print_detailed_results(results, aic_optimal, bic_optimal)
    
    print("\n\n2. SAMPLE SIZE EFFECT DEMONSTRATION")
    print("-" * 50)
    demo.demonstrate_sample_size_effect()
    
    print("\n\n3. DIFFERENT SCENARIOS DEMONSTRATION")
    print("-" * 50)
    scenario_results = demo.demonstrate_different_scenarios()
    
    print("\n\n" + "="*80)
    print("THEORETICAL BACKGROUND")
    print("="*80)
    print("""
AIC (Akaike's Information Criterion):
• Formula: AIC = 2k - 2ln(L) ≈ n×ln(MSE) + 2k
• Penalty: 2k (constant regardless of sample size)
• Philosophy: Minimize information loss
• Tends to select more complex models
• Asymptotically optimal for prediction

BIC (Bayesian Information Criterion):
• Formula: BIC = k×ln(n) - 2ln(L) ≈ n×ln(MSE) + 1/2×k×ln(n)
• Penalty: 1/2×k×ln(n) (increases with sample size)
• Philosophy: Find the true model
• Tends to select simpler models
• Consistent (selects true model as n→∞)

Key Differences:
• BIC penalty grows with sample size, AIC penalty is constant
• BIC is more conservative (prefers simpler models)
• AIC optimizes predictive performance, BIC seeks true model
• For large n: BIC penalty >> AIC penalty
• Choice depends on goal: prediction (AIC) vs interpretation (BIC)
    """)
    
    print("\n" + "="*80)
    print("DEMO COMPLETED - Check generated PNG files for visualizations!")
    print("="*80)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs('Lecture_8_Method_Selection', exist_ok=True)
    
    # Run the demonstration
    main()

