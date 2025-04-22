import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# Create a function to generate datasets
def generate_datasets():
    # Dataset 1: 2D points for visualization
    X1 = np.random.randn(100, 2) * 0.5 + np.array([2, 2])
    X2 = np.random.randn(100, 2) * 0.5 + np.array([4, 4])
    
    # Dataset 2: Correlated data for Mahalanobis distance
    cov_matrix = np.array([[2.0, 1.5], [1.5, 2.0]])
    correlated_data = np.random.multivariate_normal(
        mean=[3, 3], cov=cov_matrix, size=200)
    
    # Dataset 3: Binary vectors for Jaccard similarity
    binary_data1 = np.random.randint(0, 2, size=(10, 20))
    binary_data2 = np.random.randint(0, 2, size=(10, 20))
    
    # Dataset 4: Point sets for Hausdorff distance
    set_A = np.random.rand(15, 2) * 3
    set_B = np.random.rand(20, 2) * 3 + np.array([1, 1])
    
    return {
        'points_2d': np.vstack([X1, X2]),
        'labels_2d': np.hstack([np.zeros(100), np.ones(100)]),
        'correlated_data': correlated_data,
        'binary_data1': binary_data1,
        'binary_data2': binary_data2,
        'set_A': set_A,
        'set_B': set_B
    }

# 1. Minkowski Distance (Lp Norm)
def plot_minkowski_distances():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Minkowski Distance (Lp Norm)', fontsize=16)
    
    # Define two points for demonstration
    point1 = np.array([1, 1])
    point2 = np.array([4, 5])
    
    # Create a grid of points
    x = np.linspace(-1, 6, 100)
    y = np.linspace(-1, 6, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Calculate distances for different p values
    p_values = [1, 2, 3, np.inf]
    titles = ['Manhattan (L1)', 'Euclidean (L2)', 'L3 Norm', 'Chebyshev (L∞)']
    
    for i, (p, title) in enumerate(zip(p_values, titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Calculate Minkowski distance from point1 to all grid points
        if p == np.inf:
            distances = np.array([distance.chebyshev(point1, point) for point in grid_points])
        else:
            distances = np.array([distance.minkowski(point1, point, p) for point in grid_points])
        
        # Reshape distances for contour plot
        Z = distances.reshape(X.shape)
        
        # Plot contour lines
        contour = ax.contour(X, Y, Z, levels=10, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Plot the two points
        ax.scatter(point1[0], point1[1], color='red', s=100, label='Point 1')
        ax.scatter(point2[0], point2[1], color='blue', s=100, label='Point 2')
        
        # Calculate and display the distance between the two points
        if p == np.inf:
            dist = distance.chebyshev(point1, point2)
        else:
            dist = distance.minkowski(point1, point2, p)
        
        # For Manhattan distance, show the path
        if p == 1:
            ax.plot([point1[0], point1[0], point2[0]], 
                    [point1[1], point2[1], point2[1]], 
                    'r--', linewidth=2)
        # For Euclidean, show the direct line
        elif p == 2:
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r--', linewidth=2)
        
        ax.set_title(f'{title} Distance: {dist:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# 2. Mahalanobis Distance
def plot_mahalanobis_distance(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Mahalanobis Distance vs Euclidean Distance', fontsize=16)
    
    # Calculate mean and covariance
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    
    # Create a grid for visualization
    x = np.linspace(min(data[:, 0]) - 1, max(data[:, 0]) + 1, 100)
    y = np.linspace(min(data[:, 1]) - 1, max(data[:, 1]) + 1, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Calculate Euclidean distances from mean to all grid points
    euclidean_distances = np.array([distance.euclidean(mean, point) for point in grid_points])
    Z_euclidean = euclidean_distances.reshape(X.shape)
    
    # Calculate Mahalanobis distances from mean to all grid points
    inv_cov = np.linalg.inv(cov)
    mahalanobis_distances = np.array([distance.mahalanobis(point, mean, inv_cov) for point in grid_points])
    Z_mahalanobis = mahalanobis_distances.reshape(X.shape)
    
    # Plot Euclidean distance contours
    contour1 = ax1.contour(X, Y, Z_euclidean, levels=10, cmap='viridis')
    ax1.clabel(contour1, inline=True, fontsize=8)
    ax1.scatter(data[:, 0], data[:, 1], alpha=0.5, color='blue')
    ax1.scatter(mean[0], mean[1], color='red', s=100, marker='x', label='Mean')
    ax1.set_title('Euclidean Distance')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Mahalanobis distance contours
    contour2 = ax2.contour(X, Y, Z_mahalanobis, levels=10, cmap='viridis')
    ax2.clabel(contour2, inline=True, fontsize=8)
    ax2.scatter(data[:, 0], data[:, 1], alpha=0.5, color='blue')
    ax2.scatter(mean[0], mean[1], color='red', s=100, marker='x', label='Mean')
    
    # Plot the covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 95% confidence ellipse
    ellipse = patches.Ellipse(mean, width, height, angle=angle, 
                             fill=False, color='red', linestyle='--', linewidth=2)
    ax2.add_patch(ellipse)
    
    ax2.set_title('Mahalanobis Distance')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# 3. Cosine Similarity
def plot_cosine_similarity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cosine Similarity', fontsize=16)
    
    # Create vectors for demonstration
    vector1 = np.array([1, 1])
    vector2 = np.array([0, 2])
    vector3 = np.array([-1, 1])
    vector4 = np.array([-1, -1])
    
    # Plot vectors in 2D
    ax1.quiver(0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
    ax1.quiver(0, 0, vector2[0], vector2[1], angles='xy', scale_units='xy', scale=1, color='g', label='Vector 2')
    ax1.quiver(0, 0, vector3[0], vector3[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector 3')
    ax1.quiver(0, 0, vector4[0], vector4[1], angles='xy', scale_units='xy', scale=1, color='purple', label='Vector 4')
    
    # Calculate and display cosine similarities
    cos_sim_12 = cosine_similarity([vector1], [vector2])[0][0]
    cos_sim_13 = cosine_similarity([vector1], [vector3])[0][0]
    cos_sim_14 = cosine_similarity([vector1], [vector4])[0][0]
    
    ax1.text(0.5, -0.1, f"cos(v1,v2) = {cos_sim_12:.2f}", transform=ax1.transAxes)
    ax1.text(0.5, -0.2, f"cos(v1,v3) = {cos_sim_13:.2f}", transform=ax1.transAxes)
    ax1.text(0.5, -0.3, f"cos(v1,v4) = {cos_sim_14:.2f}", transform=ax1.transAxes)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Vector Representation')
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Create a 3D visualization of cosine similarity
    # Generate points on a unit sphere
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Reference vector for cosine similarity
    ref_vector = np.array([1, 0, 0])
    ref_vector = ref_vector / np.linalg.norm(ref_vector)
    
    # Calculate cosine similarity for each point on the sphere
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    similarities = np.array([np.dot(point, ref_vector) for point in points])
    similarities = similarities.reshape(x.shape)
    
    # Plot the sphere with cosine similarity coloring
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x, y, z, facecolors=plt.cm.viridis(0.5 * (similarities + 1)), alpha=0.8)
    
    # Plot the reference vector
    ax2.quiver(0, 0, 0, ref_vector[0], ref_vector[1], ref_vector[2], 
               color='red', linewidth=3, label='Reference Vector')
    
    ax2.set_title('Cosine Similarity on Unit Sphere')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_box_aspect([1, 1, 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# 4. Jaccard Coefficient
def plot_jaccard_similarity(binary_data1, binary_data2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Jaccard Similarity', fontsize=16)
    
    # Select two binary vectors for visualization
    vec1 = binary_data1[0]
    vec2 = binary_data2[0]
    
    # Calculate Jaccard similarity
    intersection = np.sum(np.logical_and(vec1, vec2))
    union = np.sum(np.logical_or(vec1, vec2))
    jaccard_sim = intersection / union if union > 0 else 0
    
    # Visualize the binary vectors
    ax1.imshow([vec1, vec2], cmap='binary', aspect='auto')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Vector 1', 'Vector 2'])
    ax1.set_title(f'Binary Vectors (Jaccard Similarity = {jaccard_sim:.2f})')
    ax1.set_xlabel('Features')
    
    # Create a Venn diagram-like visualization
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Draw two circles representing sets
    circle1 = plt.Circle((4, 5), 3, alpha=0.5, color='blue', label='Set A')
    circle2 = plt.Circle((6, 5), 3, alpha=0.5, color='red', label='Set B')
    
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)
    
    # Calculate areas for the Venn diagram
        # Calculate areas for the Venn diagram
    total_ones_vec1 = np.sum(vec1)
    total_ones_vec2 = np.sum(vec2)
    
    # Add text annotations to the Venn diagram
    ax2.text(3, 5, f"A: {total_ones_vec1}", fontsize=12, ha='center')
    ax2.text(7, 5, f"B: {total_ones_vec2}", fontsize=12, ha='center')
    ax2.text(5, 5, f"A∩B: {intersection}", fontsize=12, ha='center', 
             bbox=dict(facecolor='white', alpha=0.7))
    
    ax2.set_title(f'Jaccard Similarity = |A∩B|/|A∪B| = {jaccard_sim:.2f}')
    ax2.set_xlabel('Jaccard Distance = 1 - Jaccard Similarity = {:.2f}'.format(1 - jaccard_sim))
    ax2.set_aspect('equal')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax2.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# 5. Hausdorff Distance
def plot_hausdorff_distance(set_A, set_B):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Hausdorff Distance', fontsize=16)
    
    # Calculate Hausdorff distance
    def directed_hausdorff(A, B):
        # Calculate distances from each point in A to all points in B
        distances = np.array([[distance.euclidean(a, b) for b in B] for a in A])
        # Find the minimum distance for each point in A
        min_distances = np.min(distances, axis=1)
        # Return the maximum of these minimum distances
        return np.max(min_distances), np.argmax(min_distances)
    
    # Calculate directed Hausdorff distances
    h_AB, idx_A = directed_hausdorff(set_A, set_B)
    h_BA, idx_B = directed_hausdorff(set_B, set_A)
    
    # Hausdorff distance is the maximum of the two directed distances
    hausdorff_dist = max(h_AB, h_BA)
    
    # Plot the point sets
    ax.scatter(set_A[:, 0], set_A[:, 1], color='blue', label='Set A', s=50)
    ax.scatter(set_B[:, 0], set_B[:, 1], color='red', label='Set B', s=50)
    
    # Highlight the points that determine the Hausdorff distance
    if h_AB >= h_BA:
        # Point in A that is farthest from its nearest point in B
        point_A = set_A[idx_A]
        # Find the nearest point in B to this point
        distances_to_B = [distance.euclidean(point_A, b) for b in set_B]
        idx_nearest_B = np.argmin(distances_to_B)
        point_B = set_B[idx_nearest_B]
        
        ax.scatter(point_A[0], point_A[1], color='blue', s=200, edgecolor='black', 
                   linewidth=2, label='Max min point in A')
        ax.scatter(point_B[0], point_B[1], color='red', s=200, edgecolor='black', 
                   linewidth=2, label='Nearest point in B')
        
        # Draw a line connecting these points
        ax.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], 'k--', linewidth=2)
    else:
        # Point in B that is farthest from its nearest point in A
        point_B = set_B[idx_B]
        # Find the nearest point in A to this point
        distances_to_A = [distance.euclidean(point_B, a) for a in set_A]
        idx_nearest_A = np.argmin(distances_to_A)
        point_A = set_A[idx_nearest_A]
        
        ax.scatter(point_B[0], point_B[1], color='red', s=200, edgecolor='black', 
                   linewidth=2, label='Max min point in B')
        ax.scatter(point_A[0], point_A[1], color='blue', s=200, edgecolor='black', 
                   linewidth=2, label='Nearest point in A')
        
        # Draw a line connecting these points
        ax.plot([point_B[0], point_A[0]], [point_B[1], point_A[1]], 'k--', linewidth=2)
    
    # Add text explaining the Hausdorff distance
    ax.text(0.05, 0.95, f"Hausdorff distance: {hausdorff_dist:.2f}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.text(0.05, 0.85, f"h(A,B): {h_AB:.2f}", transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.text(0.05, 0.75, f"h(B,A): {h_BA:.2f}", transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.set_title('Hausdorff Distance = max(h(A,B), h(B,A))')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# Main function to run all visualizations
def main():
    print("Generating datasets...")
    datasets = generate_datasets()
    
    print("Plotting Minkowski distances...")
    fig_minkowski = plot_minkowski_distances()
    fig_minkowski.savefig('minkowski_distances.png', dpi=300, bbox_inches='tight')
    
    print("Plotting Mahalanobis distance...")
    fig_mahalanobis = plot_mahalanobis_distance(datasets['correlated_data'])
    fig_mahalanobis.savefig('mahalanobis_distance.png', dpi=300, bbox_inches='tight')
    
    print("Plotting Cosine similarity...")
    fig_cosine = plot_cosine_similarity()
    fig_cosine.savefig('cosine_similarity.png', dpi=300, bbox_inches='tight')
    
    print("Plotting Jaccard similarity...")
    fig_jaccard = plot_jaccard_similarity(datasets['binary_data1'], datasets['binary_data2'])
    fig_jaccard.savefig('jaccard_similarity.png', dpi=300, bbox_inches='tight')
    
    print("Plotting Hausdorff distance...")
    fig_hausdorff = plot_hausdorff_distance(datasets['set_A'], datasets['set_B'])
    fig_hausdorff.savefig('hausdorff_distance.png', dpi=300, bbox_inches='tight')
    
    print("All visualizations completed and saved!")
    plt.show()

if __name__ == "__main__":
    main()

