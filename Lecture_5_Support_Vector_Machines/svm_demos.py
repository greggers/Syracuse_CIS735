import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

def plot_svm_decision_function(X, y, clf, ax=None, plot_support=True):
    """
    Plot the decision function for a 2D SVM classifier.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        The input data
    y : array-like, shape (n_samples,)
        The target values
    clf : sklearn.svm.SVC
        The fitted SVM classifier
    ax : matplotlib.axes.Axes, optional
        The axes to plot on
    plot_support : bool, default=True
        Whether to plot support vectors
    """
    if ax is None:
        ax = plt.gca()
    
    # Plot the data points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis', zorder=10, edgecolors='k')
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Get the decision function values
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and margins
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'], zorder=1)
    
    # Plot the decision function as a filled contour
    contour = ax.contourf(xx, yy, Z, cmap='RdBu', alpha=0.3, zorder=0)
    
    # Plot support vectors if requested
    if plot_support and hasattr(clf, 'support_vectors_'):
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=200, linewidth=1, facecolors='none', edgecolors='black',
                   zorder=10, label='Support Vectors')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    return contour

def linear_svm_demo():
    """
    Demonstrate a linear SVM on a linearly separable dataset.
    """
    # Generate a linearly separable dataset
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    
    # Make it more linearly separable by stretching the data
    transformation = np.array([[1, 0], [0, 2]])
    X = np.dot(X, transformation)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train a linear SVM
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Linear SVM on Linearly Separable Data', fontsize=16)
    
    # Plot the decision function on the training data
    contour = plot_svm_decision_function(X_train, y_train, clf, ax=ax1)
    ax1.set_title(f'Training Data with Decision Boundary\nAccuracy: {accuracy:.2f}')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    
    # Plot the decision function on all data
    plot_svm_decision_function(X, y, clf, ax=ax2)
    ax2.set_title('Full Dataset with Decision Boundary')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    
    # Add a colorbar for the decision function
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label('Decision Function Value')
    
    # Add some information about the SVM
    n_sv = clf.support_vectors_.shape[0]
    w = clf.coef_[0]
    b = clf.intercept_[0]
    margin = 2 / np.linalg.norm(w)
    
    info_text = (
        f"Number of support vectors: {n_sv}\n"
        f"Weight vector: [{w[0]:.2f}, {w[1]:.2f}]\n"
        f"Bias term: {b:.2f}\n"
        f"Margin width: {margin:.2f}"
    )
    
    ax2.text(0.05, 0.05, info_text, transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('linear_svm_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return clf, X, y

def nonlinear_svm_demo():
    """
    Demonstrate a non-linear SVM using polynomial kernel on data that cannot be
    separated with a linear kernel.
    """
    # Generate a dataset that is not linearly separable
    # Create a circular pattern where points inside the circle belong to one class
    # and points outside belong to another class
    n_samples = 200
    np.random.seed(42)
    
    # Generate points in a circle
    radius = 5.0
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.sqrt(np.random.uniform(0, radius**2, n_samples))
    
    # Convert to Cartesian coordinates
    X = np.zeros((n_samples, 2))
    X[:, 0] = r * np.cos(theta)
    X[:, 1] = r * np.sin(theta)
    
    # Assign labels: points inside a smaller circle are class 1, outside are class 0
    inner_circle_radius = 2.5
    y = (r < inner_circle_radius).astype(int)
    
    # Add some noise
    noise_level = 0.3
    noise = np.random.normal(0, noise_level, X.shape)
    X += noise
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train a linear SVM
    linear_clf = svm.SVC(kernel='linear', C=1.0)
    linear_clf.fit(X_train, y_train)
    
    # Create and train a polynomial SVM with degree=2
    poly_clf = svm.SVC(kernel='poly', degree=2, C=1.0, gamma='scale')
    poly_clf.fit(X_train, y_train)
    
    # Evaluate both models
    linear_pred = linear_clf.predict(X_test)
    poly_pred = poly_clf.predict(X_test)
    
    linear_accuracy = accuracy_score(y_test, linear_pred)
    poly_accuracy = accuracy_score(y_test, poly_pred)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Linear vs Polynomial Kernel SVM on Non-Linearly Separable Data', fontsize=16)
    
    # Plot the decision function for the linear SVM
    contour1 = plot_svm_decision_function(X_train, y_train, linear_clf, ax=ax1)
    ax1.set_title(f'Linear Kernel (p=1)\nTest Accuracy: {linear_accuracy:.2f}')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    
    # Plot the decision function for the polynomial SVM
    contour2 = plot_svm_decision_function(X_train, y_train, poly_clf, ax=ax2)
    ax2.set_title(f'Polynomial Kernel (p=2)\nTest Accuracy: {poly_accuracy:.2f}')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    
    # Add colorbars for the decision functions
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('Decision Function Value')
    
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('Decision Function Value')
    
    # Add information about the SVMs
    linear_sv_count = linear_clf.support_vectors_.shape[0]
    poly_sv_count = poly_clf.support_vectors_.shape[0]
    
    linear_info = f"Support vectors: {linear_sv_count}\nKernel: linear"
    poly_info = f"Support vectors: {poly_sv_count}\nKernel: polynomial (degree=2)"
    
    ax1.text(0.05, 0.05, linear_info, transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.text(0.05, 0.05, poly_info, transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('nonlinear_svm_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return linear_clf, poly_clf, X, y

def rbf_kernel_demo():
    """
    Demonstrate the effect of different gamma values in RBF kernel SVM.
    Gamma is inversely proportional to sigma (width of the Gaussian).
    """
    # Generate a dataset with a complex decision boundary
    n_samples = 200
    np.random.seed(42)
    
    # Create a moon-shaped dataset
    X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    
    # Scale the data to make visualization clearer
    X = X * 2
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a figure with multiple subplots for different gamma values
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Effect of Gamma Parameter in RBF Kernel SVM', fontsize=16)
    
    # Test different gamma values (inversely proportional to sigma²)
    gamma_values = [0.1, 0.5, 1.0, 5.0]
    axes = axes.flatten()
    
    for i, gamma in enumerate(gamma_values):
        # Create and train an RBF SVM with the current gamma value
        clf = svm.SVC(kernel='rbf', gamma=gamma, C=10.0)
        clf.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Plot the decision function
        contour = plot_svm_decision_function(X_train, y_train, clf, ax=axes[i])
        axes[i].set_title(f'Gamma = {gamma} (σ ≈ {1/np.sqrt(2*gamma):.2f})\nAccuracy: {accuracy:.2f}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        
        # Add information about support vectors
        n_sv = clf.support_vectors_.shape[0]
        axes[i].text(0.05, 0.95, f"Support vectors: {n_sv}", 
                     transform=axes[i].transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
        
        # Add a colorbar for the decision function
        plt.colorbar(contour, ax=axes[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('rbf_kernel_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a figure to compare RBF with linear and polynomial kernels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparison of Different SVM Kernels on Non-Linear Data', fontsize=16)
    
    # Train models with different kernels
    linear_clf = svm.SVC(kernel='linear', C=10.0)
    linear_clf.fit(X_train, y_train)
    linear_acc = accuracy_score(y_test, linear_clf.predict(X_test))
    
    poly_clf = svm.SVC(kernel='poly', degree=2, C=10.0, gamma='scale')
    poly_clf.fit(X_train, y_train)
    poly_acc = accuracy_score(y_test, poly_clf.predict(X_test))
    
    rbf_clf = svm.SVC(kernel='rbf', gamma=1.0, C=10.0)
    rbf_clf.fit(X_train, y_train)
    rbf_acc = accuracy_score(y_test, rbf_clf.predict(X_test))
    
    # Plot the decision functions
    plot_svm_decision_function(X_train, y_train, linear_clf, ax=axes[0])
    axes[0].set_title(f'Linear Kernel\nAccuracy: {linear_acc:.2f}')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    plot_svm_decision_function(X_train, y_train, poly_clf, ax=axes[1])
    axes[1].set_title(f'Polynomial Kernel (degree=2)\nAccuracy: {poly_acc:.2f}')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    
    plot_svm_decision_function(X_train, y_train, rbf_clf, ax=axes[2])
    axes[2].set_title(f'RBF Kernel (gamma=1.0)\nAccuracy: {rbf_acc:.2f}')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('kernel_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rbf_clf, X, y

def compare_c_values():
    """
    Compare different C values for a linear SVM.
    """
    # Generate a linearly separable dataset with some noise
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    transformation = np.array([[1, 0], [0, 2]])
    X = np.dot(X, transformation)
    
    # Add some noise points
    np.random.seed(42)
    noise_points = 5
    noise_indices = np.random.choice(len(X), noise_points, replace=False)
    y[noise_indices] = 1 - y[noise_indices]  # Flip labels for noise points
    
    # Create a figure with multiple subplots for different C values
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Effect of C Parameter in Linear SVM', fontsize=16)
    
    c_values = [0.1, 1.0, 10.0, 100.0]
    axes = axes.flatten()
    
    for i, C in enumerate(c_values):
        # Create and train a linear SVM with the current C value
        clf = svm.SVC(kernel='linear', C=C)
        clf.fit(X, y)
        
        # Plot the decision function
        plot_svm_decision_function(X, y, clf, ax=axes[i])
        axes[i].set_title(f'C = {C}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        
        # Add information about support vectors
        n_sv = clf.support_vectors_.shape[0]
        axes[i].text(0.05, 0.95, f"Support vectors: {n_sv}", 
                     transform=axes[i].transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('svm_c_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Running Linear SVM Demo...")
    clf, X, y = linear_svm_demo()
    
    print("\nDemonstrating Linear vs Polynomial Kernel SVM...")
    linear_clf, poly_clf, X_nonlinear, y_nonlinear = nonlinear_svm_demo()
    
    print("\nDemonstrating RBF Kernel with different gamma values...")
    rbf_clf, X_rbf, y_rbf = rbf_kernel_demo()
    
    print("\nComparing different C values...")
    compare_c_values()
