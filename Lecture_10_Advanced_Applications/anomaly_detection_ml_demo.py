import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import uuid

class AnomalyDetectionDigitalThread:
    """
    ML-powered anomaly detection with complete digital thread tracking
    """
    
    def __init__(self):
        self.thread_events = []
        self.models = {}
        self.detection_results = []
        self.feature_importance = {}
        
    def log_event(self, event_type: str, details: dict, model_info: dict = None):
        """Log events in the digital thread"""
        event = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'model_info': model_info or {}
        }
        self.thread_events.append(event)
        return event['event_id']
    
    def generate_network_traffic_data(self, n_samples=1000, anomaly_rate=0.05):
        """Generate synthetic network traffic data with known anomalies"""
        np.random.seed(42)
        
        # Normal traffic patterns
        normal_samples = int(n_samples * (1 - anomaly_rate))
        
        # Features: bytes_per_second, packets_per_second, connection_duration, unique_ports
        normal_data = np.random.multivariate_normal(
            mean=[1000, 50, 30, 5],
            cov=[[200000, 1000, 500, 10],
                 [1000, 100, 20, 2],
                 [500, 20, 100, 1],
                 [10, 2, 1, 4]],
            size=normal_samples
        )
        
        # Anomalous traffic (DDoS, data exfiltration, port scanning)
        anomaly_samples = n_samples - normal_samples
        
        # DDoS attacks - high bytes and packets
        ddos_data = np.random.multivariate_normal(
            mean=[10000, 500, 5, 3],
            cov=[[5000000, 10000, 100, 5],
                 [10000, 2500, 50, 2],
                 [100, 50, 25, 1],
                 [5, 2, 1, 2]],
            size=anomaly_samples // 3
        )
        
        # Data exfiltration - high bytes, low packets, long duration
        exfil_data = np.random.multivariate_normal(
            mean=[5000, 10, 300, 2],
            cov=[[1000000, 100, 5000, 2],
                 [100, 25, 50, 1],
                 [5000, 50, 10000, 5],
                 [2, 1, 5, 1]],
            size=anomaly_samples // 3
        )
        
        # Port scanning - low bytes, high unique ports
        scan_data = np.random.multivariate_normal(
            mean=[100, 100, 1, 50],
            cov=[[10000, 500, 10, 100],
                 [500, 2500, 20, 200],
                 [10, 20, 1, 5],
                 [100, 200, 5, 625]],
            size=anomaly_samples - (anomaly_samples // 3) * 2
        )
        
        # Combine all data
        X = np.vstack([normal_data, ddos_data, exfil_data, scan_data])
        y = np.hstack([
            np.zeros(normal_samples),
            np.ones(len(ddos_data)),
            np.ones(len(exfil_data)) * 2,
            np.ones(len(scan_data)) * 3
        ])
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=['bytes_per_sec', 'packets_per_sec', 'duration', 'unique_ports'])
        df['anomaly_type'] = y
        df['is_anomaly'] = (y > 0).astype(int)
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
        
        # Ensure non-negative values
        df[['bytes_per_sec', 'packets_per_sec', 'duration', 'unique_ports']] = \
            df[['bytes_per_sec', 'packets_per_sec', 'duration', 'unique_ports']].abs()
        
        return df
    
    def train_anomaly_detection_model(self, data: pd.DataFrame):
        """Train ML model for anomaly detection with thread tracking"""
        
        # Log data preparation
        data_event_id = self.log_event(
            'data_preparation',
            {
                'total_samples': len(data),
                'features': ['bytes_per_sec', 'packets_per_sec', 'duration', 'unique_ports'],
                'known_anomalies': data['is_anomaly'].sum(),
                'anomaly_rate': data['is_anomaly'].mean()
            }
        )
        
        # Feature scaling
        features = ['bytes_per_sec', 'packets_per_sec', 'duration', 'unique_ports']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[features])
        
        scaling_event_id = self.log_event(
            'feature_scaling',
            {
                'scaling_method': 'StandardScaler',
                'feature_means': scaler.mean_.tolist(),
                'feature_stds': scaler.scale_.tolist()
            }
        )
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=0.1,  # Expected proportion of anomalies
            random_state=42,
            n_estimators=100
        )
        
        model.fit(X_scaled)
        
        # Calculate feature importance using permutation
        feature_importance = self._calculate_feature_importance(model, X_scaled, features)
        
        training_event_id = self.log_event(
            'model_training',
            {
                'algorithm': 'IsolationForest',
                'contamination': 0.1,
                'n_estimators': 100,
                'training_samples': len(X_scaled)
            },
            model_info={
                'model_type': 'anomaly_detection',
                'feature_importance': feature_importance
            }
        )
        
        # Store model and scaler
        self.models['anomaly_detector'] = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'training_event_id': training_event_id
        }
        
        self.feature_importance = feature_importance
        
        return model, scaler
    
    def _calculate_feature_importance(self, model, X, feature_names):
        """Calculate feature importance using permutation method"""
        baseline_scores = model.decision_function(X)
        baseline_anomalies = (baseline_scores < 0).sum()
        
        importance = {}
        for i, feature in enumerate(feature_names):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            permuted_scores = model.decision_function(X_permuted)
            permuted_anomalies = (permuted_scores < 0).sum()
            
            # Importance is the change in anomaly detection rate
            importance[feature] = abs(baseline_anomalies - permuted_anomalies) / len(X)
        
        return importance
    
    def real_time_anomaly_detection(self, new_data: pd.DataFrame):
        """Perform real-time anomaly detection with thread tracking"""
        
        if 'anomaly_detector' not in self.models:
            raise ValueError("Model not trained yet!")
        
        model_info = self.models['anomaly_detector']
        model = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features']
        
        # Preprocess new data
        X_new = scaler.transform(new_data[features])
        
        # Predict anomalies
        anomaly_scores = model.decision_function(X_new)
        predictions = model.predict(X_new)  # -1 for anomalies, 1 for normal
        
        # Convert predictions to binary (1 for anomaly, 0 for normal)
        is_anomaly = (predictions == -1).astype(int)
        
        # Log detection results
        detection_results = []
        for i, (score, pred, is_anom) in enumerate(zip(anomaly_scores, predictions, is_anomaly)):
            if is_anom:
                # Classify type of anomaly based on feature values
                anomaly_type = self._classify_anomaly_type(new_data.iloc[i])
                
                result = {
                    'timestamp': new_data.iloc[i]['timestamp'],
                    'anomaly_score': float(score),
                    'is_anomaly': bool(is_anom),
                    'anomaly_type': anomaly_type,
                    'confidence': float(abs(score)),
                    'features': new_data.iloc[i][features].to_dict()
                }
                
                detection_results.append(result)
                
                # Log individual anomaly detection
                self.log_event(
                    'anomaly_detected',
                    {
                        'anomaly_score': float(score),
                        'anomaly_type': anomaly_type,
                        'confidence': float(abs(score)),
                        'affected_features': new_data.iloc[i][features].to_dict()
                    },
                    model_info={
                        'model_used': 'anomaly_detector',
                        'detection_threshold': 0.0
                    }
                )
        
        # Log batch detection summary
        batch_event_id = self.log_event(
            'batch_anomaly_detection',
            {
                'samples_processed': len(new_data),
                'anomalies_detected': len(detection_results),
                'detection_rate': len(detection_results) / len(new_data),
                'avg_anomaly_score': float(np.mean(anomaly_scores[is_anomaly == 1])) if np.any(is_anomaly) else 0.0
            }
        )
        
        self.detection_results.extend(detection_results)
        return detection_results
    
    def _classify_anomaly_type(self, sample):
        """Classify the type of anomaly based on feature patterns"""
        bytes_per_sec = sample['bytes_per_sec']
        packets_per_sec = sample['packets_per_sec']
        duration = sample['duration']
        unique_ports = sample['unique_ports']
        
        # DDoS: High bytes and packets, short duration
        if bytes_per_sec > 5000 and packets_per_sec > 200 and duration < 10:
            return 'ddos_attack'
        
        # Data exfiltration: High bytes, low packets, long duration
        elif bytes_per_sec > 3000 and packets_per_sec < 30 and duration > 100:
            return 'data_exfiltration'
        
        # Port scanning: Many unique ports, low bytes
        elif unique_ports > 20 and bytes_per_sec < 500:
            return 'port_scanning'
        
        else:
            return 'unknown_anomaly'
    
    def evaluate_model_performance(self, test_data: pd.DataFrame):
        """Evaluate model performance with thread tracking"""
        
        if 'anomaly_detector' not in self.models:
            raise ValueError("Model not trained yet!")
        
        # Get predictions
        detection_results = self.real_time_anomaly_detection(test_data)
        
        # Create prediction array
        y_pred = np.zeros(len(test_data))
        for result in detection_results:
            # Find matching timestamp
            mask = test_data['timestamp'] == result['timestamp']
            y_pred[mask] = 1
        
        y_true = test_data['is_anomaly'].values
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Log evaluation results
        eval_event_id = self.log_event(
            'model_evaluation',
            {
                'test_samples': len(test_data),
                'true_anomalies': int(y_true.sum()),
                'detected_anomalies': int(y_pred.sum()),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            },
            model_info={
                'evaluation_method': 'holdout_test_set',
                'metrics_used': ['precision', 'recall', 'f1_score']
            }
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def visualize_anomaly_detection(self, data: pd.DataFrame, detection_results: list):
        """Visualize anomaly detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Time series of anomalies
        axes[0, 0].scatter(data['timestamp'], data['bytes_per_sec'], 
                          c=data['is_anomaly'], cmap='coolwarm', alpha=0.6)
        axes[0, 0].set_title('Anomalies in Network Traffic (Bytes/sec)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Bytes per Second')
        
        # Feature importance
        features = list(self.feature_importance.keys())
        importance_values = list(self.feature_importance.values())
        axes[0, 1].bar(features, importance_values)
        axes[0, 1].set_title('Feature Importance for Anomaly Detection')
        axes[0, 1].set_ylabel('Importance Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Anomaly types distribution
        if detection_results:
            anomaly_types = [r['anomaly_type'] for r in detection_results]
            type_counts = pd.Series(anomaly_types).value_counts()
            axes[1, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Distribution of Anomaly Types')
        
        # Anomaly scores distribution
        if detection_results:
            scores = [r['anomaly_score'] for r in detection_results]
            axes[1, 1].hist(scores, bins=20, alpha=0.7, color='red')
            axes[1, 1].set_title('Distribution of Anomaly Scores')
            axes[1, 1].set_xlabel('Anomaly Score')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def generate_anomaly_report(self, filename: str = "anomaly_detection_report.json"):
        """Generate comprehensive anomaly detection report"""
        report = {
            'report_generated': datetime.now().isoformat(),
            'summary': {
                'total_events': len(self.thread_events),
                'total_anomalies_detected': len(self.detection_results),
                'model_performance': self._get_latest_performance_metrics(),
                'feature_importance': self.feature_importance
            },
            'digital_thread': self.thread_events,
            'detection_results': self.detection_results,
            'model_lineage': self._get_model_lineage()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Anomaly detection report exported to {filename}")
        return filename
    
    def _get_latest_performance_metrics(self):
        """Get the latest model performance metrics from thread"""
        for event in reversed(self.thread_events):
            if event['event_type'] == 'model_evaluation':
                return {
                    'precision': event['details']['precision'],
                    'recall': event['details']['recall'],
                    'f1_score': event['details']['f1_score']
                }
        return {}
    
    def _get_model_lineage(self):
        """Get model training and evaluation lineage"""
        lineage = []
        for event in self.thread_events:
            if event['event_type'] in ['model_training', 'model_evaluation', 'feature_scaling']:
                lineage.append({
                    'timestamp': event['timestamp'],
                    'event_type': event['event_type'],
                    'details': event['details']
                })
        return lineage

# Demo execution
if __name__ == "__main__":
    print("=== ML-Powered Anomaly Detection Digital Thread Demo ===\n")
    
    # Initialize anomaly detection system
    anomaly_system = AnomalyDetectionDigitalThread()
    
    # Generate training data
    print("Step 1: Generating network traffic data...")
    training_data = anomaly_system.generate_network_traffic_data(n_samples=1000, anomaly_rate=0.05)
    
    # Train model
    print("Step 2: Training anomaly detection model...")
    model, scaler = anomaly_system.train_anomaly_detection_model(training_data)
    
    # Generate test data for real-time detection
    print("Step 3: Generating test data for real-time detection...")
    test_data = anomaly_system.generate_network_traffic_data(n_samples=200, anomaly_rate=0.08)
    
    # Perform real-time detection
    print("Step 4: Performing real-time anomaly detection...")
    detection_results = anomaly_system.real_time_anomaly_detection(test_data)
    
    # Evaluate performance
    print("Step 5: Evaluating model performance...")
    performance = anomaly_system.evaluate_model_performance(test_data)
    
    # Display results
    print(f"\n=== Detection Results ===")
    print(f"Anomalies detected: {len(detection_results)}")
    print(f"Detection rate: {len(detection_results)/len(test_data):.2%}")
    print(f"Model Performance:")
    print(f"  Precision: {performance['precision']:.3f}")
    print(f"  Recall: {performance['recall']:.3f}")
    print(f"  F1-Score: {performance['f1_score']:.3f}")
    
    # Show anomaly types
    if detection_results:
        anomaly_types = {}
        for result in detection_results:
            atype = result['anomaly_type']
            anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
        
        print(f"\n=== Anomaly Types Detected ===")
        for atype, count in anomaly_types.items():
            print(f"  {atype}: {count}")
    
    # Visualize results
    anomaly_system.visualize_anomaly_detection(test_data, detection_results)
    
    # Generate report
    report_file = anomaly_system.generate_anomaly_report()
    
    print(f"\n=== Digital Thread Summary ===")
    print(f"Total thread events: {len(anomaly_system.thread_events)}")
    print(f"Feature importance: {anomaly_system.feature_importance}")
