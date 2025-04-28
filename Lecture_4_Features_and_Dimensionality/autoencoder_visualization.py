import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.manifold import TSNE

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the standard autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim=32):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.encoder(x)

# Function to train a model
def train_model(model, train_loader, num_epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            
            # Forward pass
            output = model(img)
            loss = criterion(output, img.view(img.size(0), -1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

# Function to add noise to images
def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0, 1)

# Part 1: Standard Autoencoder for Image Compression
# -------------------------------------------------
encoding_dim = 32
autoencoder = Autoencoder(encoding_dim).to(device)
train_model(autoencoder, train_loader, num_epochs=10)

# Select a few test images for visualization
num_images = 5
test_iter = iter(test_loader)
test_images, _ = next(test_iter)
test_images = test_images[:num_images].to(device)

# Encode and decode the test images
autoencoder.eval()
with torch.no_grad():
    decoded_images = autoencoder(test_images)
    encoded_features = autoencoder.encode(test_images)

# Convert tensors to numpy arrays for plotting
test_images_np = test_images.cpu().numpy()
decoded_images_np = decoded_images.view(num_images, 28, 28).cpu().numpy()
encoded_features_np = encoded_features.cpu().numpy()

# Create figure for standard autoencoder visualization
plt.figure(figsize=(15, 10))
plt.suptitle('Standard Autoencoder for Image Compression', fontsize=16)

# Plot original images
for i in range(num_images):
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(test_images_np[i, 0], cmap='gray')
    plt.title(f"Original {i+1}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Visualize the encoded representations
for i in range(num_images):
    ax = plt.subplot(3, num_images, i + 1 + num_images)
    
    # Create a 2D visualization of the encoding
    encoded_img = encoded_features_np[i]
    
    # Reshape to a square-like grid for visualization
    grid_size = int(np.ceil(np.sqrt(encoding_dim)))
    encoded_grid = np.zeros((grid_size, grid_size))
    for j in range(encoding_dim):
        row, col = j // grid_size, j % grid_size
        encoded_grid[row, col] = encoded_img[j]
    
    plt.imshow(encoded_grid, cmap='viridis')
    plt.title(f"Encoded {i+1} ({encoding_dim} values)")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Plot reconstructed images
for i in range(num_images):
    ax = plt.subplot(3, num_images, i + 1 + 2*num_images)
    plt.imshow(decoded_images_np[i], cmap='gray')
    plt.title(f"Reconstructed {i+1}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('standard_autoencoder.png', dpi=300, bbox_inches='tight')

# Part 2: Denoising Autoencoder
# ----------------------------
# Define the denoising autoencoder (same architecture as standard autoencoder)
denoising_autoencoder = Autoencoder(encoding_dim).to(device)

# Create a noisy training dataset
def train_denoising_autoencoder(model, train_loader, noise_factor=0.5, num_epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            
            # Add noise to the input images
            noisy_img = add_noise(img, noise_factor)
            
            # Forward pass (noisy input, clean target)
            output = model(noisy_img)
            loss = criterion(output, img.view(img.size(0), -1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

# Train the denoising autoencoder
train_denoising_autoencoder(denoising_autoencoder, train_loader, noise_factor=0.5, num_epochs=10)

# Create noisy test images
noisy_test_images = add_noise(test_images, noise_factor=0.5)

# Denoise the test images
denoising_autoencoder.eval()
with torch.no_grad():
    denoised_images = denoising_autoencoder(noisy_test_images)

# Convert tensors to numpy arrays for plotting
noisy_test_images_np = noisy_test_images.cpu().numpy()
denoised_images_np = denoised_images.view(num_images, 28, 28).cpu().numpy()

# Create figure for denoising autoencoder visualization
plt.figure(figsize=(15, 10))
plt.suptitle('Denoising Autoencoder', fontsize=16)

# Plot original clean images
for i in range(num_images):
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(test_images_np[i, 0], cmap='gray')
    plt.title(f"Original {i+1}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Plot noisy images
for i in range(num_images):
    ax = plt.subplot(3, num_images, i + 1 + num_images)
    plt.imshow(noisy_test_images_np[i, 0], cmap='gray')
    plt.title(f"Noisy {i+1}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Plot denoised images
for i in range(num_images):
    ax = plt.subplot(3, num_images, i + 1 + 2*num_images)
    plt.imshow(denoised_images_np[i], cmap='gray')
    plt.title(f"Denoised {i+1}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Add explanatory text
plt.figtext(0.5, 0.01, 
            "Denoising autoencoders learn to remove noise from corrupted input data.\n"
            "They are trained to map noisy examples to clean targets, forcing the network\n"
            "to learn robust features that capture the underlying structure of the data.",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('denoising_autoencoder.png', dpi=300, bbox_inches='tight')

# Create a figure to compare compression ratios
plt.figure(figsize=(12, 6))

# Try different encoding dimensions
encoding_dims = [8, 16, 32, 64, 128]
compression_ratios = [784/dim for dim in encoding_dims]
reconstruction_errors = []

# Function to calculate mean squared error
def mse(x, y):
    return ((x - y) ** 2).mean().item()

# Train models with different encoding dimensions and measure reconstruction error
for dim in encoding_dims:
    print(f"Training autoencoder with encoding dimension {dim}...")
    
    # Define a simple autoencoder with the current encoding dimension
    temp_autoencoder = Autoencoder(encoding_dim=dim).to(device)
    
    # Train the model (fewer epochs for quicker results)
    train_model(temp_autoencoder, train_loader, num_epochs=5)
    
    # Calculate reconstruction error on a batch of test data
    temp_autoencoder.eval()
    with torch.no_grad():
        test_batch, _ = next(iter(test_loader))
        test_batch = test_batch.to(device)
        reconstructed = temp_autoencoder(test_batch)
        error = mse(reconstructed, test_batch.view(test_batch.size(0), -1))
        reconstruction_errors.append(error)

# Plot compression ratio vs. reconstruction error
plt.subplot(1, 2, 1)
plt.plot(encoding_dims, reconstruction_errors, 'o-', linewidth=2)
plt.title('Encoding Dimension vs. Reconstruction Error')
plt.xlabel('Encoding Dimension')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)
plt.xscale('log', base=2)

plt.subplot(1, 2, 2)
plt.plot(compression_ratios, reconstruction_errors, 'o-', linewidth=2)
plt.title('Compression Ratio vs. Reconstruction Error')
plt.xlabel('Compression Ratio (784/dim)')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('compression_analysis.png', dpi=300, bbox_inches='tight')

plt.show()
