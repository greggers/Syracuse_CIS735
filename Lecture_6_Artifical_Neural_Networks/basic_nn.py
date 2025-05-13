import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class BasicNeuralNetwork(nn.Module):
    """
    A basic multi-layer neural network with configurable architecture.
    
    Parameters:
    -----------
    input_size : int
        Number of input features
    hidden_sizes : list of int
        List containing the number of neurons in each hidden layer
    output_size : int
        Number of output classes
    activation : torch.nn.Module, default=nn.ReLU()
        Activation function to use between layers
    dropout_rate : float, default=0.2
        Dropout probability for regularization
    """
    def __init__(self, input_size, hidden_sizes, output_size, 
                 activation=nn.ReLU(), dropout_rate=0.2):
        super(BasicNeuralNetwork, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation)
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # If binary classification, add sigmoid activation
        if output_size == 1:
            layers.append(nn.Sigmoid())
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, epochs=100, early_stopping_patience=10):
    """
    Train the neural network model
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimization algorithm
    device : torch.device
        Device to run the training on (CPU or GPU)
    epochs : int, default=100
        Number of training epochs
    early_stopping_patience : int, default=10
        Number of epochs to wait for improvement before stopping
        
    Returns:
    --------
    dict
        Dictionary containing training and validation losses and accuracies
    """
    model.to(device)
    
    # For tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Reshape targets for binary classification if needed
            if outputs.shape[1] == 1:
                targets = targets.float().view(-1, 1)
                loss = criterion(outputs, targets)
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Reshape targets for binary classification if needed
                if outputs.shape[1] == 1:
                    targets = targets.float().view(-1, 1)
                    loss = criterion(outputs, targets)
                    predicted = (outputs > 0.5).float()
                else:
                    loss = criterion(outputs, targets)
                    _, predicted = torch.max(outputs, 1)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / train_total
        epoch_val_loss = val_loss / val_total
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total
        
        # Store metrics
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train Loss: {epoch_train_loss:.4f} | '
                  f'Val Loss: {epoch_val_loss:.4f} | '
                  f'Train Acc: {epoch_train_acc:.4f} | '
                  f'Val Acc: {epoch_val_acc:.4f}')
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return history


def plot_decision_boundary(model, X, y, device, title="Neural Network Decision Boundary"):
    """
    Plot the decision boundary for a 2D dataset
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained neural network model
    X : numpy.ndarray
        Input features (2D)
    y : numpy.ndarray
        Target labels
    device : torch.device
        Device the model is on
    title : str, default="Neural Network Decision Boundary"
        Plot title
    """
    model.eval()
    
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Convert to PyTorch tensors and make predictions
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    
    with torch.no_grad():
        outputs = model(grid_tensor)
        
        if outputs.shape[1] == 1:  # Binary classification
            Z = (outputs > 0.5).cpu().numpy().reshape(xx.shape)
        else:  # Multi-class
            Z = torch.argmax(outputs, dim=1).cpu().numpy().reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', 
                         cmap=plt.cm.coolwarm, alpha=0.8)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    plt.tight_layout()
    
    return plt


def plot_training_history(history):
    """
    Plot the training and validation metrics
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    return plt


def run_binary_classification_example():
    """Run a binary classification example with the neural network"""
    print("Running binary classification example...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate a non-linear dataset (moons)
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define the model (binary classification)
    model = BasicNeuralNetwork(
        input_size=2,
        hidden_sizes=[16, 8],
        output_size=1,  # Binary classification
        dropout_rate=0.1
    )
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=100
    )
    
    # Plot training history
    history_plot = plot_training_history(history)
    history_plot.savefig('binary_nn_training_history.png')
    
    # Plot decision boundary
    decision_plot = plot_decision_boundary(
        model=model,
        X=X,
        y=y,
        device=device,
        title="Neural Network Decision Boundary (Binary Classification)"
    )
    decision_plot.savefig('binary_nn_decision_boundary.png')
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        test_predicted = (test_outputs > 0.5).float().cpu().numpy().flatten()
        test_accuracy = np.mean(test_predicted == y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")
    
    plt.show()


def run_multiclass_classification_example():
    """Run a multi-class classification example with the neural network"""
    print("Running multi-class classification example...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate a multi-class dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_informative=2, 
        n_redundant=0, 
        n_classes=3, 
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define the model (multi-class classification)
    model = BasicNeuralNetwork(
        input_size=2,
        hidden_sizes=[32, 16],
        output_size=3,  # 3 classes
        dropout_rate=0.1
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=100
    )
    
    # Plot training history
    history_plot = plot_training_history(history)
    history_plot.savefig('multiclass_nn_training_history.png')
    
    # Plot decision boundary
    decision_plot = plot_decision_boundary(
        model=model,
        X=X,
        y=y,
        device=device,
        title="Neural Network Decision Boundary (Multi-class Classification)"
    )
    decision_plot.savefig('multiclass_nn_decision_boundary.png')
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(X_test_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
        total = y_test_tensor.size(0)
        correct = (predicted.cpu() == y_test_tensor).sum().item()
        test_accuracy = correct / total
        print(f"Test accuracy: {test_accuracy:.4f}")
    
    plt.show()


def compare_architectures():
    """Compare different neural network architectures on the same dataset"""
    print("Comparing different neural network architectures...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate a non-linear dataset (moons)
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define different architectures to compare
    architectures = [
        {"name": "Shallow (1 layer)", "hidden_sizes": [8]},
        {"name": "Medium (2 layers)", "hidden_sizes": [16, 8]},
        {"name": "Deep (3 layers)", "hidden_sizes": [32, 16, 8]},
        {"name": "Wide (1 wide layer)", "hidden_sizes": [64]},
    ]
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Compare architectures
    results = []
    
    plt.figure(figsize=(15, 10))
    
    for i, arch in enumerate(architectures):
        print(f"Training {arch['name']} architecture...")
        
        # Create model with the current architecture
        model = BasicNeuralNetwork(
            input_size=2,
            hidden_sizes=arch["hidden_sizes"],
            output_size=1,  # Binary classification
            dropout_rate=0.1
        )
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Train the model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=100
        )
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(device))
            test_predicted = (test_outputs > 0.5).float().cpu().numpy().flatten()
            test_accuracy = np.mean(test_predicted == y_test)
        
        results.append({
            "name": arch["name"],
            "test_accuracy": test_accuracy,
            "history": history
        })
        
        # Plot decision boundary for this architecture
        plt.subplot(2, 2, i+1)
        
        # Create a mesh grid
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Convert to PyTorch tensors and make predictions
        grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
        
        with torch.no_grad():
            outputs = model(grid_tensor)
            Z = (outputs > 0.5).cpu().numpy().reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        
        # Plot the data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', 
                             cmap=plt.cm.coolwarm, alpha=0.8)
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"{arch['name']} - Accuracy: {test_accuracy:.4f}")
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png')
    
    # Plot training curves for all architectures
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    for result in results:
        plt.plot(result["history"]["val_loss"], label=result["name"])
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot(result["history"]["val_acc"], label=result["name"])
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('architecture_training_comparison.png')
    
    # Print results summary
    print("\nArchitecture Comparison Results:")
    print("-" * 50)
    for result in results:
        print(f"{result['name']}: Test Accuracy = {result['test_accuracy']:.4f}")
    
    plt.show()

def run_cnn_example():
    """Run a Convolutional Neural Network example on MNIST dataset"""
    print("Running CNN example on MNIST dataset...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Import torchvision for datasets
    import torchvision
    import torchvision.transforms as transforms
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Define CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # First convolutional layer
            # Input: 1x28x28, Output: 32x26x26
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
            # Second convolutional layer
            # Input: 32x13x13 (after pooling), Output: 64x11x11
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
            # Max pooling layer
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # Fully connected layers
            # Input: 64x5x5 (after pooling), flattened to 1600
            self.fc1 = nn.Linear(64 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 10)
            # Dropout for regularization
            self.dropout = nn.Dropout(0.25)
            # Activation function
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # First conv block
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            
            # Second conv block
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            
            # Flatten the output for the fully connected layers
            x = x.view(-1, 64 * 5 * 5)
            
            # Fully connected layers
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    # Initialize the CNN
    model = CNN()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Training loop
    num_epochs = 5
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Track statistics
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_acc = correct / total
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} | '
              f'Test Loss: {epoch_test_loss:.4f} | '
              f'Train Acc: {epoch_acc:.4f} | '
              f'Test Acc: {epoch_test_acc:.4f}')
    
    # Plot training and testing curves
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Testing Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cnn_training_curves.png')
    
    # Visualize CNN filters
    def visualize_filters(model):
        # Get the weights from the first convolutional layer
        weights = model.conv1.weight.data.cpu().numpy()
        
        # Create a figure to display the filters
        fig, axes = plt.subplots(4, 8, figsize=(12, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Plot each filter
        for i, ax in enumerate(axes.flat):
            if i < weights.shape[0]:  # Only plot if filter exists
                # Get the filter
                img = weights[i, 0]
                
                # Normalize filter for better visualization
                img = (img - img.min()) / (img.max() - img.min())
                
                # Display the filter
                ax.imshow(img, cmap='viridis')
                ax.set_title(f'Filter {i+1}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.suptitle('First Convolutional Layer Filters')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.savefig('cnn_filters.png')
    
    # Visualize some sample predictions
    def visualize_predictions(model, test_loader, device, num_samples=10):
        model.eval()
        
        # Get a batch of test images
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(images[:num_samples].to(device))
            _, predicted = torch.max(outputs, 1)
        
        # Convert images for display
        images = images[:num_samples].cpu().numpy()
        
        # Create a figure to display the images and predictions
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.subplots_adjust(hspace=0.5)
        
        # Plot each image with its prediction
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                # Display the image
                ax.imshow(images[i].squeeze(), cmap='gray')
                
                # Get the true and predicted labels
                true_label = labels[i].item()
                pred_label = predicted[i].cpu().item()
                
                # Set the title with true and predicted labels
                title_color = 'green' if true_label == pred_label else 'red'
                ax.set_title(f'True: {true_label}\nPred: {pred_label}', 
                             color=title_color)
                ax.axis('off')
        
        plt.suptitle('CNN Predictions on MNIST Test Images')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.savefig('cnn_predictions.png')
    
    # Visualize filters and predictions
    visualize_filters(model)
    visualize_predictions(model, test_loader, device)
    
    # Final test accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    final_accuracy = correct / total
    print(f"Final test accuracy: {final_accuracy:.4f}")
    
    plt.show()


if __name__ == "__main__":
    print("Basic Neural Network Demonstrations")
    print("=" * 40)
    
    # Run binary classification example
    run_binary_classification_example()
    
    # Run multi-class classification example
    run_multiclass_classification_example()
    
    # Compare different architectures
    compare_architectures()
    
    # Run CNN example
    run_cnn_example()


