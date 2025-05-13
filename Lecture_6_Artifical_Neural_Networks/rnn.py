import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import math
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleRNN(nn.Module):
    """
    Basic RNN model for time series forecasting
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    """
    GRU model for time series forecasting
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer (not a parameter, but should be part of the module's state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    """
    Transformer model for time series forecasting
    """
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, output_size):
        super(TransformerModel, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_size)
    
    def forward(self, src):
        # Project input to d_model dimensions
        src = self.input_projection(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src)
        
        # Get output from the last time step
        output = self.output_layer(output[:, -1, :])
        
        return output


def generate_time_series_data(n_samples=1000):
    """
    Generate a synthetic time series dataset with multiple patterns
    """
    # Time steps
    time = np.arange(0, n_samples)
    
    # Generate components
    trend = 0.001 * time**2
    seasonal = 5 * np.sin(2 * np.pi * time / 365.25)
    noise = 0.5 * np.random.randn(n_samples)
    
    # Combine components
    data = trend + seasonal + noise
    
    return data


def prepare_time_series_data(data, seq_length, train_size=0.8):
    """
    Prepare time series data for training and testing
    """
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_normalized) - seq_length):
        X.append(data_normalized[i:i+seq_length])
        y.append(data_normalized[i+seq_length])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, y_train, X_test, y_test, scaler


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, learning_rate=0.001):
    """
    Train a model and return training history and time
    """
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # For tracking metrics
    train_losses = []
    test_losses = []
    
    # Start timer
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(device)
            y_test_device = y_test.to(device)
            
            test_outputs = model(X_test_device)
            test_loss = criterion(test_outputs, y_test_device).item()
            test_losses.append(test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}')
    
    # End timer
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_time': training_time
    }


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model and return predictions and metrics
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        X_test_device = X_test.to(device)
        predictions = model(X_test_device).cpu().numpy()
    
    # Inverse transform predictions and actual values
    y_test_np = y_test.numpy()
    
    # Reshape for inverse transform
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test_np)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    
    return predictions_rescaled, y_test_rescaled, rmse


def plot_training_history(histories, model_names):
    """
    Plot training and testing loss for multiple models
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for i, model_name in enumerate(model_names):
        plt.plot(histories[i]['train_losses'], label=f'{model_name}')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plot testing loss
    plt.subplot(1, 2, 2)
    for i, model_name in enumerate(model_names):
        plt.plot(histories[i]['test_losses'], label=f'{model_name}')
    plt.title('Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('rnn_training_history.png')


def plot_predictions(predictions_list, y_test, model_names, scaler, data, seq_length):
    """
    Plot predictions from multiple models
    """
    plt.figure(figsize=(15, 10))
    
    # Plot full time series
    plt.subplot(2, 1, 1)
    plt.plot(data, label='Original Time Series', alpha=0.5)
    plt.title('Full Time Series')
    plt.legend()
    
    # Plot predictions
    plt.subplot(2, 1, 2)
    
    # Get the original time indices for the test set
    test_indices = np.arange(len(data) - len(y_test), len(data))
    
    # Plot actual test values
    plt.plot(test_indices, y_test, label='Actual', marker='o', markersize=3, linestyle='-')
    
    # Plot predictions from each model
    for i, model_name in enumerate(model_names):
        plt.plot(test_indices, predictions_list[i], label=f'{model_name} Predictions', 
                 marker='.', markersize=2, linestyle='--')
    
    plt.title('Model Predictions vs Actual Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('rnn_predictions.png')


def compare_models_table(histories, rmse_values, model_names):
    """
    Create a comparison table of model performance
    """
    # Create a DataFrame for comparison
    comparison_data = {
        'Model': model_names,
        'Training Time (s)': [history['training_time'] for history in histories],
        'Final Train Loss': [history['train_losses'][-1] for history in histories],
        'Final Test Loss': [history['test_losses'][-1] for history in histories],
        'RMSE': rmse_values
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print the comparison table
    print("\nModel Comparison:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    # Save to CSV
    comparison_df.to_csv('rnn_model_comparison.csv', index=False)
    
    return comparison_df


def run_time_series_example():
    """
    Run a complete time series forecasting example comparing different RNN architectures
    """
    print("Running time series forecasting example with different RNN architectures...")
    
    # Generate synthetic time series data
    n_samples = 1500
    data = generate_time_series_data(n_samples)
    
    # Sequence length (lookback window)
    seq_length = 30
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_time_series_data(data, seq_length)
    
    print(f"Data shape - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Model parameters
    input_size = 1  # Single feature (univariate time series)
    hidden_size = 64
    output_size = 1  # Predict next value
    num_layers = 2
    
    # Transformer specific parameters
    d_model = 64
    nhead = 4
    dim_feedforward = 256
    
    # Training parameters
    epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    # Initialize models
    lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    gru_model = GRUModel(input_size, hidden_size, output_size, num_layers)
    transformer_model = TransformerModel(input_size, d_model, nhead, num_layers, dim_feedforward, output_size)
    
    # Model names for plotting
    model_names = ['LSTM', 'GRU', 'Transformer']
    models = [lstm_model, gru_model, transformer_model]
    
    # Train and evaluate each model
    histories = []
    predictions_list = []
    rmse_values = []
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\nTraining {name} model...")
        
        # Train model
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        histories.append(history)
        
        # Evaluate model
        predictions, y_test_rescaled, rmse = evaluate_model(model, X_test, y_test, scaler)
        predictions_list.append(predictions)
        rmse_values.append(rmse)
        
        print(f"{name} - RMSE: {rmse:.4f}, Training time: {history['training_time']:.2f} seconds")
    
    # Plot training history
    plot_training_history(histories, model_names)
    
    # Plot predictions
    plot_predictions(predictions_list, y_test_rescaled, model_names, scaler, data, seq_length)
    
    # Compare models
    comparison_df = compare_models_table(histories, rmse_values, model_names)
    
    # Visualize the architecture differences
    def visualize_architecture_differences():
        """Create a visual explanation of the different architectures"""
        plt.figure(figsize=(15, 10))
        
        # LSTM cell
        plt.subplot(2, 2, 1)
        lstm_img = plt.imread('https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png')
        plt.imshow(lstm_img)
        plt.title('LSTM Architecture')
        plt.axis('off')
        
        # GRU cell
        plt.subplot(2, 2, 2)
        gru_img = plt.imread('https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png')
        plt.imshow(gru_img)
        plt.title('GRU Architecture')
        plt.axis('off')
        
        # Transformer architecture
        plt.subplot(2, 2, 3)
        transformer_img = plt.imread('https://www.tensorflow.org/images/tutorials/transformer/transformer.png')
        plt.imshow(transformer_img)
        plt.title('Transformer Architecture')
        plt.axis('off')
        
        # Key differences text
        plt.subplot(2, 2, 4)
        plt.axis('off')
        differences = """
        Key Differences:
        
        LSTM:
        - Uses cell state as memory
        - Has forget, input, and output gates
        - Good at capturing long-term dependencies
        - More parameters than GRU
        
        GRU:
        - Simpler architecture than LSTM
        - Uses update and reset gates
        - Often trains faster than LSTM
        - Similar performance with fewer parameters
        
        Transformer:
        - Uses self-attention mechanism
        - Processes entire sequence at once (parallel)
        - No recurrence, captures dependencies via attention
        - Often better for very long sequences
        - Typically requires more data to train effectively
        """
        plt.text(0.1, 0.1, differences, fontsize=12, va='top')
        
        plt.tight_layout()
        plt.savefig('rnn_architecture_comparison.png')
    
    # Try to visualize architectures (may fail if images can't be loaded)
    try:
        visualize_architecture_differences()
    except Exception as e:
        print(f"Could not create architecture visualization: {e}")
        
        # Create a text-based comparison instead
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        architecture_text = """
        LSTM (Long Short-Term Memory):
        - Architecture: Input Gate, Forget Gate, Output Gate, Cell State
        - Purpose: Designed to address vanishing gradient problem in standard RNNs
        - Memory: Maintains a cell state that can store information for long periods
        - Advantages: Excellent for long-term dependencies, stable training
        - Disadvantages: More complex, more parameters, slower training
        
        GRU (Gated Recurrent Unit):
        - Architecture: Update Gate, Reset Gate
        - Purpose: Simplified version of LSTM with comparable performance
        - Memory: Uses hidden state directly without separate cell state
        - Advantages: Fewer parameters, faster training, often similar performance to LSTM
        - Disadvantages: May not capture some complex patterns that LSTM can
        
        Transformer:
        - Architecture: Self-Attention, Multi-Head Attention, Feed-Forward Networks
        - Purpose: Process entire sequences in parallel without recurrence
        - Memory: Uses attention mechanism to weigh importance of all positions
        - Advantages: Highly parallelizable, excellent for very long sequences
        - Disadvantages: Requires more data, computationally intensive for long sequences
        
        When to use each:
        - LSTM: When you need proven performance on sequential data with long dependencies
        - GRU: When you want faster training with similar performance to LSTM
        - Transformer: When you have lots of data and need to capture complex relationships
          across the entire sequence
        """
        plt.text(0.05, 0.95, architecture_text, fontsize=12, va='top')
        plt.tight_layout()
        plt.savefig('rnn_architecture_text_comparison.png')
    
    # Analyze the results
    best_model_idx = np.argmin(rmse_values)
    best_model_name = model_names[best_model_idx]
    
    print(f"\nAnalysis:")
    print(f"Best performing model: {best_model_name} with RMSE of {rmse_values[best_model_idx]:.4f}")
    
    # Compare training times
    training_times = [history['training_time'] for history in histories]
    fastest_model_idx = np.argmin(training_times)
    fastest_model_name = model_names[fastest_model_idx]
    
    print(f"Fastest model to train: {fastest_model_name} ({training_times[fastest_model_idx]:.2f} seconds)")
    
    # Compare convergence
    final_losses = [history['test_losses'][-1] for history in histories]
    best_convergence_idx = np.argmin(final_losses)
    best_convergence_name = model_names[best_convergence_idx]
    
    print(f"Model with best convergence: {best_convergence_name} (final test loss: {final_losses[best_convergence_idx]:.6f})")
    
    # Plot a focused view of the predictions for a smaller time window
    plt.figure(figsize=(12, 6))
    window_size = min(100, len(y_test_rescaled))
    start_idx = len(y_test_rescaled) - window_size
    
    test_indices = np.arange(len(data) - window_size, len(data))
    
    plt.plot(test_indices, y_test_rescaled[-window_size:], label='Actual', marker='o', markersize=4, linestyle='-')
    
    for i, model_name in enumerate(model_names):
        plt.plot(test_indices, predictions_list[i][-window_size:], label=f'{model_name}', 
                 marker='.', markersize=3, linestyle='--')
    
    plt.title('Detailed View of Model Predictions (Last 100 Time Steps)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rnn_detailed_predictions.png')
    
    return comparison_df


if __name__ == "__main__":
    print("Recurrent Neural Network Architectures Comparison")
    print("=" * 50)
    
    # Run time series example
    comparison_results = run_time_series_example()
    
    print("\nExperiment complete. Results saved to CSV and visualizations saved as PNG files.")

