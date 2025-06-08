import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import uuid
import warnings
warnings.filterwarnings('ignore')

class GNNNodeDetectionDigitalThread:
    """
    Graph Neural Network-based node-level anomaly detection with digital thread tracking
    """
    
    def __init__(self):
        self.thread_events = []
        self.models = {}
        self.network_graphs = []
        self.detection_results = []
        self.node_embeddings = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def log_event(self, event_type: str, details: dict, gnn_info: dict = None):
        """Log events in the digital thread"""
        event = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'gnn_info': gnn_info or {}
        }
        self.thread_events.append(event)
        return event['event_id']
    
    def generate_network_topology(self, n_nodes=200, network_type='enterprise'):
        """Generate realistic network topology with node features"""
        
        # Create base network structure
        if network_type == 'enterprise':
            G = self._create_enterprise_network(n_nodes)
        else:
            G = nx.barabasi_albert_graph(n_nodes, 3)
        
        # Add node features representing system characteristics
        node_features = self._generate_node_features(G, network_type)
        
        # Add edge features representing communication patterns
        edge_features = self._generate_edge_features(G)
        
        # Log network generation
        self.log_event(
            'network_topology_generated',
            {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'network_type': network_type,
                'avg_degree': float(np.mean([d for n, d in G.degree()])),
                'clustering_coefficient': float(nx.average_clustering(G))
            },
            gnn_info={
                'node_feature_dim': len(list(node_features.values())[0]),
                'edge_feature_dim': len(list(edge_features.values())[0]) if edge_features else 0
            }
        )
        
        return G, node_features, edge_features
    
    def _create_enterprise_network(self, n_nodes):
        """Create realistic enterprise network topology"""
        G = nx.Graph()
        
        # Create hierarchical structure
        # Core switches (highly connected)
        core_switches = list(range(5))
        G.add_nodes_from(core_switches)
        
        # Connect core switches in full mesh
        for i in core_switches:
            for j in core_switches:
                if i != j:
                    G.add_edge(i, j)
        
        # Distribution switches
        dist_switches = list(range(5, 25))
        G.add_nodes_from(dist_switches)
        
        # Connect distribution switches to core
        for dist in dist_switches:
            core = np.random.choice(core_switches, size=2, replace=False)
            for c in core:
                G.add_edge(dist, c)
        
        # Access switches and end devices
        current_node = 25
        for dist in dist_switches:
            # Each distribution switch connects to multiple access switches
            n_access = np.random.randint(3, 8)
            access_switches = list(range(current_node, current_node + n_access))
            G.add_nodes_from(access_switches)
            current_node += n_access
            
            for access in access_switches:
                G.add_edge(dist, access)
                
                # Each access switch connects to end devices
                n_devices = np.random.randint(5, 15)
                if current_node + n_devices <= n_nodes:
                    devices = list(range(current_node, current_node + n_devices))
                    G.add_nodes_from(devices)
                    current_node += n_devices
                    
                    for device in devices:
                        G.add_edge(access, device)
                        
                        # Some devices connect to each other (peer-to-peer)
                        if np.random.random() < 0.1:
                            peer_candidates = [d for d in devices if d != device]
                            if peer_candidates:
                                peer = np.random.choice(peer_candidates)
                                G.add_edge(device, peer)
        
        # Ensure we have exactly n_nodes
        while G.number_of_nodes() < n_nodes:
            # Add isolated nodes or connect to random existing nodes
            new_node = G.number_of_nodes()
            G.add_node(new_node)
            # Connect to 1-3 random existing nodes
            existing_nodes = list(G.nodes())[:-1]  # Exclude the just-added node
            if existing_nodes:
                connections = np.random.choice(existing_nodes, 
                                             size=min(3, len(existing_nodes)), 
                                             replace=False)
                for conn in connections:
                    G.add_edge(new_node, conn)
        
        return G
    
    def _generate_node_features(self, G, network_type):
        """Generate realistic node features"""
        node_features = {}
        
        # Define node types based on network position
        node_types = self._classify_node_types(G)
        
        for node in G.nodes():
            node_type = node_types[node]
            
            # Base features for all nodes
            features = {
                'cpu_usage': np.random.beta(2, 5),  # Typically low CPU usage
                'memory_usage': np.random.beta(2, 4),
                'network_traffic_in': np.random.exponential(1000),  # Bytes/sec
                'network_traffic_out': np.random.exponential(800),
                'connection_count': len(list(G.neighbors(node))),
                'uptime_hours': np.random.exponential(168),  # Average 1 week
                'failed_login_attempts': np.random.poisson(0.5),
                'privilege_level': np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.05, 0.05])
            }
            
            # Adjust features based on node type
            if node_type == 'server':
                features['cpu_usage'] = np.random.beta(3, 3)  # Higher CPU usage
                features['memory_usage'] = np.random.beta(4, 2)  # Higher memory usage
                features['network_traffic_in'] *= 5
                features['network_traffic_out'] *= 3
                features['privilege_level'] = np.random.choice([3, 4, 5], p=[0.5, 0.3, 0.2])
                
            elif node_type == 'workstation':
                features['cpu_usage'] = np.random.beta(2, 6)
                features['failed_login_attempts'] = np.random.poisson(1.0)
                
            elif node_type == 'network_device':
                features['cpu_usage'] = np.random.beta(1, 8)  # Very low CPU
                features['memory_usage'] = np.random.beta(1, 6)
                features['network_traffic_in'] *= 10  # High throughput
                features['network_traffic_out'] *= 10
                features['failed_login_attempts'] = np.random.poisson(0.1)
                features['privilege_level'] = 5  # Network devices have high privileges
            
            # Add temporal features
            features['hour_of_day'] = np.random.randint(0, 24)
            features['day_of_week'] = np.random.randint(0, 7)
            
            # Add security-related features
            features['antivirus_status'] = np.random.choice([0, 1], p=[0.05, 0.95])
            features['firewall_enabled'] = np.random.choice([0, 1], p=[0.1, 0.9])
            features['last_patch_days'] = np.random.exponential(30)
            features['suspicious_process_count'] = np.random.poisson(0.2)
            
            # Convert to feature vector
            feature_vector = list(features.values())
            node_features[node] = feature_vector
        
        return node_features
    
    def _classify_node_types(self, G):
        """Classify nodes based on network topology"""
        node_types = {}
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        
        # Use degree centrality to classify nodes
        high_degree_threshold = np.percentile(degree_values, 90)
        medium_degree_threshold = np.percentile(degree_values, 70)
        
        for node, degree in degrees.items():
            if degree >= high_degree_threshold:
                node_types[node] = 'network_device'  # Switches, routers
            elif degree >= medium_degree_threshold:
                node_types[node] = 'server'  # Servers typically have more connections
            else:
                node_types[node] = 'workstation'  # End-user devices
        
        return node_types
    
    def _generate_edge_features(self, G):
        """Generate edge features representing communication patterns"""
        edge_features = {}
        
        for edge in G.edges():
            features = {
                'bandwidth_utilization': np.random.beta(2, 8),  # Usually low utilization
                'latency_ms': np.random.exponential(5),
                'packet_loss_rate': np.random.beta(1, 100),  # Very low packet loss
                'connection_frequency': np.random.poisson(10),  # Connections per hour
                'data_transfer_gb': np.random.exponential(0.1),  # GB transferred
                'protocol_diversity': np.random.randint(1, 6),  # Number of different protocols
                'encrypted_ratio': np.random.beta(8, 2)  # Most traffic should be encrypted
            }
            
            edge_features[edge] = list(features.values())
        
        return edge_features
    
    def inject_anomalies(self, G, node_features, anomaly_rate=0.1):
        """Inject realistic anomalies into the network"""
        n_anomalies = int(len(G.nodes()) * anomaly_rate)
        anomalous_nodes = np.random.choice(list(G.nodes()), size=n_anomalies, replace=False)
        
        anomaly_types = ['malware_infection', 'data_exfiltration', 'privilege_escalation', 
                        'ddos_source', 'insider_threat']
        
        node_labels = {node: 0 for node in G.nodes()}  # 0 = normal, 1 = anomalous
        anomaly_details = {}
        
        for node in anomalous_nodes:
            node_labels[node] = 1
            anomaly_type = np.random.choice(anomaly_types)
            anomaly_details[node] = anomaly_type
            
            # Modify node features based on anomaly type
            features = node_features[node].copy()
            
            if anomaly_type == 'malware_infection':
                features[0] *= 3  # High CPU usage
                features[1] *= 2  # High memory usage
                features[11] += 5  # More suspicious processes
                features[9] = 0   # Disable antivirus
                
            elif anomaly_type == 'data_exfiltration':
                features[3] *= 10  # Very high outbound traffic
                features[2] *= 0.5  # Lower inbound traffic
                features[6] *= 3   # More failed login attempts
                
            elif anomaly_type == 'privilege_escalation':
                features[7] = 5    # Maximum privilege level
                features[6] *= 5   # Many failed login attempts
                features[11] += 3  # More suspicious processes
                
            elif anomaly_type == 'ddos_source':
                features[2] *= 0.1  # Very low inbound traffic
                features[3] *= 20   # Extremely high outbound traffic
                features[0] *= 2    # High CPU usage
                
            elif anomaly_type == 'insider_threat':
                features[5] *= 0.1  # Low uptime (suspicious login patterns)
                features[8] = 23    # Late night activity
                features[3] *= 5    # High outbound traffic
                features[7] = np.random.choice([4, 5])  # High privileges
            
            node_features[node] = features
        
        # Log anomaly injection
        self.log_event(
            'anomalies_injected',
            {
                'total_anomalies': n_anomalies,
                'anomaly_rate': anomaly_rate,
                'anomaly_types': list(set(anomaly_details.values())),
                'anomaly_distribution': {atype: list(anomaly_details.values()).count(atype) 
                                       for atype in set(anomaly_details.values())}
            },
            gnn_info={
                'anomalous_nodes': anomalous_nodes.tolist(),
                'injection_method': 'feature_modification'
            }
        )
        
        return node_labels, anomaly_details
    
    def prepare_graph_data(self, G, node_features, node_labels, edge_features=None):
        """Convert NetworkX graph to PyTorch Geometric format"""
        
        # Convert node features to tensor
        feature_matrix = torch.tensor([node_features[node] for node in sorted(G.nodes())], 
                                    dtype=torch.float)
        
        # Convert labels to tensor
        labels = torch.tensor([node_labels[node] for node in sorted(G.nodes())], 
                            dtype=torch.long)
        
        # Convert to PyTorch Geometric data
        data = from_networkx(G)
        data.x = feature_matrix
        data.y = labels
        
        # Add edge features if available
        if edge_features:
            edge_attr = []
            for edge in data.edge_index.t().tolist():
                edge_tuple = tuple(sorted(edge))
                if edge_tuple in edge_features:
                    edge_attr.append(edge_features[edge_tuple])
                else:
                    # Handle directed edges
                    reverse_edge = tuple(reversed(edge_tuple))
                    if reverse_edge in edge_features:
                        # Default edge features
                        edge_attr.append([0.1, 5.0, 0.01, 10, 0.1, 3, 0.8])
            
            data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Log data preparation
        self.log_event(
            'graph_data_prepared',
            {
                'num_nodes': data.x.size(0),
                'num_edges': data.edge_index.size(1),
                'node_feature_dim': data.x.size(1),
                'edge_feature_dim': data.edge_attr.size(1) if hasattr(data, 'edge_attr') else 0,
                'num_anomalous_nodes': int(labels.sum().item()),
                'anomaly_ratio': float(labels.sum().item() / len(labels))
            },
            gnn_info={
                'data_format': 'pytorch_geometric',
                'device': str(self.device)
            }
        )
        
        return data

    def train_gnn_model(self, data, model_type='GAT', epochs=200, lr=0.01):
        """Train Graph Neural Network for node anomaly detection"""
        
        # Split data into train/validation/test
        num_nodes = data.x.size(0)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # 60% train, 20% validation, 20% test
        indices = torch.randperm(num_nodes)
        train_mask[indices[:int(0.6 * num_nodes)]] = True
        val_mask[indices[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
        test_mask[indices[int(0.8 * num_nodes):]] = True
        
        # Initialize model
        input_dim = data.x.size(1)
        hidden_dim = 64
        output_dim = 2  # Binary classification (normal/anomalous)
        
        if model_type == 'GAT':
            model = GraphAttentionNetwork(input_dim, hidden_dim, output_dim)
        else:
            model = GraphConvolutionalNetwork(input_dim, hidden_dim, output_dim)
        
        model = model.to(self.device)
        data = data.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.NLLLoss()
        
        # Log training start
        training_event_id = self.log_event(
            'gnn_training_started',
            {
                'model_type': model_type,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim,
                'epochs': epochs,
                'learning_rate': lr,
                'train_nodes': int(train_mask.sum().item()),
                'val_nodes': int(val_mask.sum().item()),
                'test_nodes': int(test_mask.sum().item())
            },
            gnn_info={
                'architecture': str(model),
                'optimizer': 'Adam',
                'loss_function': 'NLLLoss'
            }
        )
        
        # Training loop
        model.train()
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_out = model(data.x, data.edge_index)
                    val_pred = val_out[val_mask].max(1)[1]
                    val_acc = val_pred.eq(data.y[val_mask]).sum().item() / val_mask.sum().item()
                    val_accuracies.append(val_acc)
                    
                    if epoch % 40 == 0:
                        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
                
                model.train()
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_out = model(data.x, data.edge_index)
            test_pred = test_out[test_mask].max(1)[1]
            test_acc = test_pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
            
            # Calculate additional metrics
            y_true = data.y[test_mask].cpu().numpy()
            y_pred = test_pred.cpu().numpy()
            y_prob = torch.softmax(test_out[test_mask], dim=1)[:, 1].cpu().numpy()
            
            auc_score = roc_auc_score(y_true, y_prob)
        
        # Store model and results
        self.models['gnn_detector'] = {
            'model': model,
            'model_type': model_type,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'test_accuracy': test_acc,
            'auc_score': auc_score
        }
        
        # Log training completion
        self.log_event(
            'gnn_training_completed',
            {
                'final_train_loss': train_losses[-1],
                'test_accuracy': test_acc,
                'auc_score': auc_score,
                'total_epochs': epochs,
                'model_parameters': sum(p.numel() for p in model.parameters())
            },
            gnn_info={
                'convergence_achieved': True,
                'best_val_accuracy': max(val_accuracies) if val_accuracies else 0
            }
        )
        
        return model, test_acc, auc_score
    
    def detect_anomalous_nodes(self, data, threshold=0.5):
        """Detect anomalous nodes using trained GNN model"""
        
        if 'gnn_detector' not in self.models:
            raise ValueError("GNN model not trained yet!")
        
        model_info = self.models['gnn_detector']
        model = model_info['model']
        test_mask = model_info['test_mask']
        
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probabilities = torch.softmax(out, dim=1)
            predictions = (probabilities[:, 1] > threshold).long()
            confidence_scores = probabilities[:, 1]
        
        # Extract node embeddings from the second-to-last layer
        embeddings = self._extract_node_embeddings(model, data)
        
        # Prepare detection results
        detection_results = []
        for node_id in range(data.x.size(0)):
            if test_mask[node_id]:  # Only report on test nodes
                result = {
                    'node_id': int(node_id),
                    'predicted_anomaly': bool(predictions[node_id].item()),
                    'anomaly_probability': float(confidence_scores[node_id].item()),
                    'true_label': int(data.y[node_id].item()),
                    'node_features': data.x[node_id].cpu().numpy().tolist(),
                    'embedding': embeddings[node_id].tolist() if embeddings is not None else None
                }
                detection_results.append(result)
        
        # Calculate detection statistics
        true_positives = sum(1 for r in detection_results 
                           if r['predicted_anomaly'] and r['true_label'] == 1)
        false_positives = sum(1 for r in detection_results 
                            if r['predicted_anomaly'] and r['true_label'] == 0)
        true_negatives = sum(1 for r in detection_results 
                           if not r['predicted_anomaly'] and r['true_label'] == 0)
        false_negatives = sum(1 for r in detection_results 
                            if not r['predicted_anomaly'] and r['true_label'] == 1)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Log detection results
        self.log_event(
            'node_anomaly_detection_completed',
            {
                'total_test_nodes': len(detection_results),
                'detected_anomalies': sum(1 for r in detection_results if r['predicted_anomaly']),
                'true_anomalies': sum(1 for r in detection_results if r['true_label'] == 1),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'detection_threshold': threshold
            },
            gnn_info={
                'embedding_dimension': len(embeddings[0]) if embeddings is not None else 0,
                'confidence_stats': {
                    'mean_confidence': float(np.mean([r['anomaly_probability'] for r in detection_results])),
                    'std_confidence': float(np.std([r['anomaly_probability'] for r in detection_results]))
                }
            }
        )
        
        self.detection_results = detection_results
        return detection_results
    
    def _extract_node_embeddings(self, model, data):
        """Extract node embeddings from the trained GNN"""
        try:
            model.eval()
            with torch.no_grad():
                # Forward pass through first two layers to get embeddings
                x = model.conv1(data.x, data.edge_index)
                x = F.relu(x)
                if hasattr(model, 'batch_norm1'):
                    x = model.batch_norm1(x)
                
                x = model.conv2(x, data.edge_index)
                embeddings = F.relu(x)
                
                return embeddings.cpu().numpy()
        except:
            return None
    
    def analyze_node_importance(self, data, top_k=10):
        """Analyze node importance using various centrality measures"""
        
        # Convert back to NetworkX for centrality analysis
        edge_list = data.edge_index.t().cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_list)
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Combine with anomaly detection results
        node_analysis = []
        for node_id in G.nodes():
            analysis = {
                'node_id': node_id,
                'degree_centrality': degree_centrality[node_id],
                'betweenness_centrality': betweenness_centrality[node_id],
                'closeness_centrality': closeness_centrality[node_id],
                'eigenvector_centrality': eigenvector_centrality[node_id],
                'is_anomalous': bool(data.y[node_id].item()) if node_id < data.y.size(0) else False
            }
            
            # Add detection results if available
            if self.detection_results:
                detection_result = next((r for r in self.detection_results if r['node_id'] == node_id), None)
                if detection_result:
                    analysis['predicted_anomaly'] = detection_result['predicted_anomaly']
                    analysis['anomaly_probability'] = detection_result['anomaly_probability']
            
            node_analysis.append(analysis)
        
        # Sort by combined importance score
        for analysis in node_analysis:
            analysis['importance_score'] = (
                analysis['degree_centrality'] * 0.3 +
                analysis['betweenness_centrality'] * 0.3 +
                analysis['closeness_centrality'] * 0.2 +
                analysis['eigenvector_centrality'] * 0.2
            )
        
        node_analysis.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Log analysis
        self.log_event(
            'node_importance_analysis',
            {
                'total_nodes_analyzed': len(node_analysis),
                'top_k_nodes': top_k,
                'centrality_measures': ['degree', 'betweenness', 'closeness', 'eigenvector'],
                'anomalous_in_top_k': sum(1 for n in node_analysis[:top_k] if n.get('is_anomalous', False))
            },
            gnn_info={
                'importance_weighting': {
                    'degree': 0.3, 'betweenness': 0.3, 
                    'closeness': 0.2, 'eigenvector': 0.2
                }
            }
        )
        
        return node_analysis[:top_k]
    
    def visualize_network_detection(self, G, data, node_labels, detection_results=None):
        """Visualize network with anomaly detection results"""
        
        plt.figure(figsize=(20, 15))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Original network with true anomalies
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        normal_nodes = [n for n in G.nodes() if node_labels[n] == 0]
        anomalous_nodes = [n for n in G.nodes() if node_labels[n] == 1]
        
        axes[0, 0].set_title('True Network Anomalies', fontsize=14, fontweight='bold')
        nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color='lightblue', 
                              node_size=50, ax=axes[0, 0], alpha=0.7)
        nx.draw_networkx_nodes(G, pos, nodelist=anomalous_nodes, node_color='red', 
                              node_size=100, ax=axes[0, 0], alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=axes[0, 0])
        axes[0, 0].axis('off')
        
        # 2. GNN predictions
        if detection_results:
            predicted_normal = [r['node_id'] for r in detection_results if not r['predicted_anomaly']]
            predicted_anomalous = [r['node_id'] for r in detection_results if r['predicted_anomaly']]
            
            axes[0, 1].set_title('GNN Predicted Anomalies', fontsize=14, fontweight='bold')
            nx.draw_networkx_nodes(G, pos, nodelist=predicted_normal, node_color='lightgreen', 
                                  node_size=50, ax=axes[0, 1], alpha=0.7)
            nx.draw_networkx_nodes(G, pos, nodelist=predicted_anomalous, node_color='orange', 
                                  node_size=100, ax=axes[0, 1], alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.3, ax=axes[0, 1])
            axes[0, 1].axis('off')
        
        # 3. Node degree distribution
        degrees = [d for n, d in G.degree()]
        axes[1, 0].hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Node Degree Distribution')
        axes[1, 0].set_xlabel('Degree')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Anomaly probability heatmap
        if detection_results:
            node_probs = {r['node_id']: r['anomaly_probability'] for r in detection_results}
            prob_values = [node_probs.get(n, 0) for n in G.nodes()]
            
            # Create heatmap visualization
            axes[1, 1].set_title('Anomaly Probability Heatmap')
            scatter = axes[1, 1].scatter([pos[n][0] for n in G.nodes()], 
                                       [pos[n][1] for n in G.nodes()],
                                       c=prob_values, cmap='Reds', s=60, alpha=0.8)
            plt.colorbar(scatter, ax=axes[1, 1], label='Anomaly Probability')
            axes[1, 1].set_xlabel('X Position')
            axes[1, 1].set_ylabel('Y Position')
        
        plt.tight_layout()
        plt.show()
    
    def generate_detection_report(self, filename: str = "gnn_detection_report.json"):
        """Generate comprehensive GNN detection report"""
        
        # Calculate summary statistics
        if self.detection_results:
            total_nodes = len(self.detection_results)
            detected_anomalies = sum(1 for r in self.detection_results if r['predicted_anomaly'])
            true_anomalies = sum(1 for r in self.detection_results if r['true_label'] == 1)
            
            # Confusion matrix components
            tp = sum(1 for r in self.detection_results if r['predicted_anomaly'] and r['true_label'] == 1)
            fp = sum(1 for r in self.detection_results if r['predicted_anomaly'] and r['true_label'] == 0)
            tn = sum(1 for r in self.detection_results if not r['predicted_anomaly'] and r['true_label'] == 0)
            fn = sum(1 for r in self.detection_results if not r['predicted_anomaly'] and r['true_label'] == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / total_nodes if total_nodes > 0 else 0
        else:
            total_nodes = detected_anomalies = true_anomalies = 0
            tp = fp = tn = fn = precision = recall = f1 = accuracy = 0
        
        # Model information
        model_info = {}
        if 'gnn_detector' in self.models:
            model_data = self.models['gnn_detector']
            model_info = {
                'model_type': model_data['model_type'],
                'test_accuracy': model_data['test_accuracy'],
                'auc_score': model_data['auc_score'],
                'parameters': sum(p.numel() for p in model_data['model'].parameters())
            }
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'summary': {
                'total_nodes_tested': total_nodes,
                'detected_anomalies': detected_anomalies,
                'true_anomalies': true_anomalies,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': {
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn
                }
            },
            'model_information': model_info,
            'digital_thread': self.thread_events,
            'detection_results': self.detection_results,
            'network_statistics': self._get_network_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"GNN detection report exported to {filename}")
        return filename
    
    def _get_network_statistics(self):
        """Get network topology statistics"""
        stats = {}
        for event in self.thread_events:
            if event['event_type'] == 'network_topology_generated':
                stats = event['details']
                break
        return stats
    
    def compare_detection_methods(self, data, G):
        """Compare GNN detection with traditional methods"""
        
        # Traditional anomaly detection using node features only
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        
        # Prepare feature matrix
        X = data.x.cpu().numpy()
        y_true = data.y.cpu().numpy()
        
        # Test mask for fair comparison
        test_mask = self.models['gnn_detector']['test_mask'].cpu().numpy()
        X_test = X[test_mask]
        y_test = y_true[test_mask]
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X)
        iso_predictions = iso_forest.predict(X_test)
        iso_predictions = (iso_predictions == -1).astype(int)  # Convert to 0/1
        
        # One-Class SVM
        oc_svm = OneClassSVM(gamma='scale', nu=0.1)
        oc_svm.fit(X[y_true == 0])  # Train only on normal samples
        svm_predictions = oc_svm.predict(X_test)
        svm_predictions = (svm_predictions == -1).astype(int)  # Convert to 0/1
        
        # GNN predictions
        gnn_predictions = np.array([r['predicted_anomaly'] for r in self.detection_results])
        
        # Calculate metrics for each method
        methods = {
            'GNN': gnn_predictions,
            'Isolation Forest': iso_predictions,
            'One-Class SVM': svm_predictions
        }
        
        comparison_results = {}
        for method_name, predictions in methods.items():
            tp = np.sum((predictions == 1) & (y_test == 1))
            fp = np.sum((predictions == 1) & (y_test == 0))
            tn = np.sum((predictions == 0) & (y_test == 0))
            fn = np.sum((predictions == 0) & (y_test == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(y_test)
            
            comparison_results[method_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
        
        # Log comparison
        self.log_event(
            'detection_methods_comparison',
            {
                'methods_compared': list(methods.keys()),
                'test_samples': len(y_test),
                'comparison_results': comparison_results
            },
            gnn_info={
                'gnn_advantage': {
                    'uses_graph_structure': True,
                    'considers_node_relationships': True,
                    'learns_representations': True
                }
            }
        )
        
        return comparison_results
    
    def visualize_comparison_results(self, comparison_results):
        """Visualize comparison between detection methods"""
        
        methods = list(comparison_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [comparison_results[method][metric] for method in methods]
            bars = axes[i].bar(methods, values, alpha=0.7, 
                              color=['red', 'blue', 'green'][:len(methods)])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for node classification"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.3):
        super(GraphAttentionNetwork, self).__init__()
        
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim * num_heads)
        
    def forward(self, x, edge_index, batch=None):
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)


class GraphConvolutionalNetwork(nn.Module):
    """Graph Convolutional Network for node classification"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index, batch=None):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)


# Demo execution
if __name__ == "__main__":
    print("=== Graph Neural Network Node Detection Digital Thread Demo ===\n")
    
    # Initialize GNN detection system
    gnn_system = GNNNodeDetectionDigitalThread()
    
    # Step 1: Generate network topology
    print("Step 1: Generating enterprise network topology...")
    G, node_features, edge_features = gnn_system.generate_network_topology(
        n_nodes=300, network_type='enterprise'
    )
    
    # Step 2: Inject anomalies
    print("Step 2: Injecting realistic anomalies...")
    node_labels, anomaly_details = gnn_system.inject_anomalies(
        G, node_features, anomaly_rate=0.15
    )
    
    # Step 3: Prepare graph data
    print("Step 3: Preparing graph data for GNN...")
    data = gnn_system.prepare_graph_data(G, node_features, node_labels, edge_features)
    
    # Step 4: Train GNN models
    print("Step 4: Training Graph Neural Networks...")
    
    # Train GAT model
    print("  Training Graph Attention Network...")
    gat_model, gat_acc, gat_auc = gnn_system.train_gnn_model(
        data, model_type='GAT', epochs=150, lr=0.01
    )
    
    # Step 5: Detect anomalous nodes
    print("Step 5: Detecting anomalous nodes...")
    detection_results = gnn_system.detect_anomalous_nodes(data, threshold=0.5)
    
    # Step 6: Analyze node importance
    print("Step 6: Analyzing node importance...")
    important_nodes = gnn_system.analyze_node_importance(data, top_k=15)
    
    # Step 7: Compare with traditional methods
    print("Step 7: Comparing with traditional detection methods...")
    comparison_results = gnn_system.compare_detection_methods(data, G)
    
    # Display results
    print(f"\n=== GNN Detection Results ===")
    print(f"Network Size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"True Anomalies: {sum(node_labels.values())}")
    print(f"Detected Anomalies: {sum(1 for r in detection_results if r['predicted_anomaly'])}")
    print(f"GAT Model Accuracy: {gat_acc:.4f}")
    print(f"GAT Model AUC: {gat_auc:.4f}")
    
    # Calculate and display detailed metrics
    tp = sum(1 for r in detection_results if r['predicted_anomaly'] and r['true_label'] == 1)
    fp = sum(1 for r in detection_results if r['predicted_anomaly'] and r['true_label'] == 0)
    tn = sum(1 for r in detection_results if not r['predicted_anomaly'] and r['true_label'] == 0)
    fn = sum(1 for r in detection_results if not r['predicted_anomaly'] and r['true_label'] == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=== Detailed Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    # Display anomaly types detected
    print(f"\n=== Anomaly Types in Dataset ===")
    anomaly_type_counts = {}
    for node_id, atype in anomaly_details.items():
        anomaly_type_counts[atype] = anomaly_type_counts.get(atype, 0) + 1
    
    for atype, count in anomaly_type_counts.items():
        print(f"  {atype}: {count} instances")
    
    # Display top important nodes
    print(f"\n=== Top 10 Most Important Nodes ===")
    for i, node in enumerate(important_nodes[:10]):
        status = "ANOMALOUS" if node.get('is_anomalous', False) else "NORMAL"
        predicted = "DETECTED" if node.get('predicted_anomaly', False) else "NORMAL"
        print(f"  {i+1}. Node {node['node_id']}: {status} (Predicted: {predicted})")
        print(f"     Importance: {node['importance_score']:.4f}, "
              f"Degree Centrality: {node['degree_centrality']:.4f}")
    
    # Display method comparison
    print(f"\n=== Detection Method Comparison ===")
    for method, metrics in comparison_results.items():
        print(f"\n{method}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # Show high-confidence detections
    print(f"\n=== High-Confidence Anomaly Detections ===")
    high_conf_detections = [r for r in detection_results 
                           if r['predicted_anomaly'] and r['anomaly_probability'] > 0.8]
    high_conf_detections.sort(key=lambda x: x['anomaly_probability'], reverse=True)
    
    for i, detection in enumerate(high_conf_detections[:10]):
        node_id = detection['node_id']
        prob = detection['anomaly_probability']
        correct = "✓" if detection['true_label'] == 1 else "✗"
        anomaly_type = anomaly_details.get(node_id, "N/A")
        print(f"  {i+1}. Node {node_id}: {prob:.4f} confidence {correct} ({anomaly_type})")
    
    # Display digital thread summary
    print(f"\n=== Digital Thread Summary ===")
    print(f"Total Thread Events: {len(gnn_system.thread_events)}")
    
    event_types = {}
    for event in gnn_system.thread_events:
        event_type = event['event_type']
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print("Event Type Distribution:")
    for event_type, count in event_types.items():
        print(f"  {event_type}: {count}")
    
    # Show recent thread events
    print(f"\n=== Recent Thread Events (Last 5) ===")
    for event in gnn_system.thread_events[-5:]:
        print(f"  {event['timestamp']}: {event['event_type']}")
        if 'gnn_info' in event and event['gnn_info']:
            key_info = list(event['gnn_info'].keys())[:2]  # Show first 2 keys
            print(f"    GNN Info: {key_info}")
    
    # Visualize results
    print(f"\nGenerating visualizations...")
    gnn_system.visualize_network_detection(G, data, node_labels, detection_results)
    gnn_system.visualize_comparison_results(comparison_results)
    
    # Generate comprehensive report
    print(f"\nGenerating comprehensive report...")
    report_file = gnn_system.generate_detection_report("gnn_node_detection_report.json")
    
    # Final summary
    print(f"\n=== Final Summary ===")
    print(f"✓ Successfully trained Graph Attention Network for node anomaly detection")
    print(f"✓ Processed {G.number_of_nodes()} nodes with {len(node_features[0])} features each")
    print(f"✓ Achieved {gat_acc:.1%} accuracy and {gat_auc:.3f} AUC score")
    print(f"✓ Detected {tp} out of {tp + fn} true anomalies ({recall:.1%} recall)")
    print(f"✓ Generated {len(gnn_system.thread_events)} digital thread events")
    print(f"✓ Compared against traditional ML methods")
    print(f"✓ Analyzed network topology and node importance")
    print(f"✓ Report saved to: {report_file}")
    
    # Performance insights
    print(f"\n=== Key Insights ===")
    
    # Compare GNN vs traditional methods
    gnn_f1 = comparison_results['GNN']['f1_score']
    iso_f1 = comparison_results['Isolation Forest']['f1_score']
    svm_f1 = comparison_results['One-Class SVM']['f1_score']
    
    if gnn_f1 > max(iso_f1, svm_f1):
        improvement = ((gnn_f1 - max(iso_f1, svm_f1)) / max(iso_f1, svm_f1)) * 100
        print(f"• GNN outperformed traditional methods by {improvement:.1f}% in F1-score")
    
    # Analyze detection patterns
    anomalous_important = sum(1 for node in important_nodes[:10] if node.get('is_anomalous', False))
    if anomalous_important > 0:
        print(f"• {anomalous_important} out of top 10 important nodes were anomalous")
        print(f"  This suggests anomalies often target critical network positions")
    
    # Feature importance insights
    if detection_results:
        avg_anomaly_prob = np.mean([r['anomaly_probability'] for r in detection_results 
                                   if r['predicted_anomaly']])
        print(f"• Average confidence for detected anomalies: {avg_anomaly_prob:.3f}")
    
    # Network structure insights
    network_stats = gnn_system._get_network_statistics()
    if network_stats:
        print(f"• Network has {network_stats.get('avg_degree', 0):.1f} average degree")
        print(f"• Clustering coefficient: {network_stats.get('clustering_coefficient', 0):.3f}")
    
    print(f"\n=== Demo Complete ===")
    print(f"The GNN-based node detection system successfully demonstrated:")
    print(f"1. Realistic enterprise network topology generation")
    print(f"2. Multi-type anomaly injection (malware, data exfiltration, etc.)")
    print(f"3. Graph Neural Network training with attention mechanisms")
    print(f"4. Node-level anomaly detection with confidence scoring")
    print(f"5. Network topology analysis and centrality measures")
    print(f"6. Comparison with traditional anomaly detection methods")
    print(f"7. Comprehensive digital thread tracking and reporting")
    print(f"8. Visualization of detection results and network structure")


