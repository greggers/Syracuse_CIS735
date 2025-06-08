import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import hashlib
import json
from datetime import datetime

class MLModelDigitalThread:
    """
    Tracks the complete digital thread of an ML model from data to deployment
    """
    
    def __init__(self):
        self.thread_log = []
        self.model_artifacts = {}
    
    def log_event(self, event_type: str, details: dict, artifacts: dict = None):
        """Log an event in the digital thread"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'artifacts': artifacts or {}
        }
        self.thread_log.append(event)
        
        if artifacts:
            self.model_artifacts.update(artifacts)
    
    def generate_data_hash(self, data):
        """Generate hash for data provenance"""
        return hashlib.sha256(str(data.values.tobytes()).encode()).hexdigest()[:16]
    
    def simulate_ml_pipeline(self):
        """Simulate complete ML pipeline with digital thread tracking"""
        
        # 1. Data Collection
        print("Step 1: Data Collection")
        # Simulate network traffic data for malware detection
        np.random.seed(42)
        n_samples = 1000
        
        # Features: packet_size, connection_duration, port_number, byte_frequency
        X = np.random.rand(n_samples, 4)
        X[:, 0] *= 1500  # packet_size
        X[:, 1] *= 300   # connection_duration
        X[:, 2] = np.random.randint(1, 65536, n_samples)  # port_number
        X[:, 3] *= 100   # byte_frequency
        
        # Labels: 0 = benign, 1 = malware
        y = (X[:, 0] > 750) & (X[:, 1] < 150) & (X[:, 3] > 50)
        y = y.astype(int)
        
        df = pd.DataFrame(X, columns=['packet_size', 'connection_duration', 
                                     'port_number', 'byte_frequency'])
        df['label'] = y
        
        data_hash = self.generate_data_hash(df)
        
        self.log_event(
            event_type="data_collection",
            details={
                "source": "network_traffic_logs",
                "samples": n_samples,
                "features": list(df.columns[:-1]),
                "data_hash": data_hash,
                "collection_method": "pcap_analysis"
            },
            artifacts={"raw_data_hash": data_hash}
        )
        
        # 2. Data Preprocessing
        print("Step 2: Data Preprocessing")
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        preprocessing_hash = hashlib.sha256(str(X_scaled.tobytes()).encode()).hexdigest()[:16]
        
        self.log_event(
            event_type="data_preprocessing",
            details={
                "transformations": ["standard_scaling"],
                "scaler_params": {
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist()
                },
                "processed_data_hash": preprocessing_hash
            },
            artifacts={"scaler": scaler, "processed_data_hash": preprocessing_hash}
        )
        
        # 3. Model Training
        print("Step 3: Model Training")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Model fingerprint
        model_params = model.get_params()
        model_hash = hashlib.sha256(str(model_params).encode()).hexdigest()[:16]
        
        self.log_event(
            event_type="model_training",
            details={
                "algorithm": "RandomForestClassifier",
                "hyperparameters": model_params,
                "training_samples": len(X_train),
                "model_hash": model_hash,
                "feature_importance": dict(zip(
                    ['packet_size', 'connection_duration', 'port_number', 'byte_frequency'],
                    model.feature_importances_.tolist()
                ))
            },
            artifacts={"model": model, "model_hash": model_hash}
        )
        
        # 4. Model Evaluation
        print("Step 4: Model Evaluation")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.log_event(
            event_type="model_evaluation",
            details={
                "test_samples": len(X_test),
                "accuracy": accuracy,
                "evaluation_method": "holdout_validation",
                "metrics": {
                    "true_positives": int(np.sum((y_test == 1) & (y_pred == 1))),
                    "false_positives": int(np.sum((y_test == 0) & (y_pred == 1))),
                    "true_negatives": int(np.sum((y_test == 0) & (y_pred == 0))),
                    "false_negatives": int(np.sum((y_test == 1) & (y_pred == 0)))
                }
            }
        )
        
        # 5. Model Deployment
        print("Step 5: Model Deployment")
        model_version = f"malware_detector_v1.0_{datetime.now().strftime('%Y%m%d')}"
        
        self.log_event(
            event_type="model_deployment",
            details={
                "deployment_target": "production_environment",
                "model_version": model_version,
                "deployment_method": "containerized_service",
                "monitoring_enabled": True,
                "rollback_available": True
            },
            artifacts={"deployment_version": model_version}
        )
        
        return model, scaler, df
    
    def get_model_provenance(self):
        """Get complete model provenance information"""
        provenance = {
            "model_lineage": [],
            "data_lineage": [],
            "artifacts": list(self.model_artifacts.keys()),
            "total_events": len(self.thread_log)
        }
        
        for event in self.thread_log:
            if event['event_type'] in ['model_training', 'model_evaluation', 'model_deployment']:
                provenance["model_lineage"].append({
                    'timestamp': event['timestamp'],
                    'event': event['event_type'],
                    'details': event['details']
                })
            elif event['event_type'] in ['data_collection', 'data_preprocessing']:
                provenance["data_lineage"].append({
                    'timestamp': event['timestamp'],
                    'event': event['event_type'],
                    'details': event['details']
                })
        
        return provenance
    
    def visualize_thread_timeline(self):
        """Visualize the digital thread timeline"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        timestamps = [datetime.fromisoformat(event['timestamp']) for event in self.thread_log]
        events = [event['event_type'] for event in self.thread_log]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {
            'data_collection': 'blue',
            'data_preprocessing': 'green',
            'model_training': 'red',
            'model_evaluation': 'orange',
            'model_deployment': 'purple'
        }
        
        for i, (timestamp, event) in enumerate(zip(timestamps, events)):
            ax.scatter(timestamp, i, c=colors.get(event, 'gray'), s=100, alpha=0.7)
            ax.annotate(event.replace('_', ' ').title(), 
                       (timestamp, i), 
                       xytext=(10, 0), 
                       textcoords='offset points',
                       fontsize=9,
                       ha='left')
        
        ax.set_yticks(range(len(events)))
        ax.set_yticklabels([f"Step {i+1}" for i in range(len(events))])
        ax.set_xlabel('Timeline')
        ax.set_title('ML Model Digital Thread Timeline')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def export_thread_report(self, filename: str = "ml_model_thread_report.json"):
        """Export complete digital thread report"""
        report = {
            "report_generated": datetime.now().isoformat(),
            "thread_summary": {
                "total_events": len(self.thread_log),
                "pipeline_duration": self._calculate_pipeline_duration(),
                "artifacts_generated": len(self.model_artifacts)
            },
            "complete_thread": self.thread_log,
            "provenance": self.get_model_provenance()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Digital thread report exported to {filename}")
        return filename
    
    def _calculate_pipeline_duration(self):
        """Calculate total pipeline duration"""
        if len(self.thread_log) < 2:
            return "0 seconds"
        
        start_time = datetime.fromisoformat(self.thread_log[0]['timestamp'])
        end_time = datetime.fromisoformat(self.thread_log[-1]['timestamp'])
        duration = end_time - start_time
        
        return str(duration)

# Demo execution
if __name__ == "__main__":
    print("=== ML Model Digital Thread Demo ===\n")
    
    # Initialize digital thread tracker
    ml_thread = MLModelDigitalThread()
    
    # Run the complete ML pipeline with tracking
    model, scaler, data = ml_thread.simulate_ml_pipeline()
    
    print(f"\n=== Digital Thread Summary ===")
    print(f"Total events tracked: {len(ml_thread.thread_log)}")
    print(f"Artifacts generated: {len(ml_thread.model_artifacts)}")
    
    # Show provenance
    provenance = ml_thread.get_model_provenance()
    print(f"\n=== Model Provenance ===")
    print("Data Lineage:")
    for item in provenance['data_lineage']:
        print(f"  - {item['timestamp']}: {item['event']}")
    
    print("\nModel Lineage:")
    for item in provenance['model_lineage']:
        print(f"  - {item['timestamp']}: {item['event']}")
    
    # Visualize timeline
    ml_thread.visualize_thread_timeline()
    
    # Export report
    report_file = ml_thread.export_thread_report()
    
    print(f"\n=== Thread Verification ===")
    print("Data Hash Chain:")
    for event in ml_thread.thread_log:
        if 'data_hash' in event['details']:
            print(f"  {event['event_type']}: {event['details']['data_hash']}")
        elif 'processed_data_hash' in event['details']:
            print(f"  {event['event_type']}: {event['details']['processed_data_hash']}")
        elif 'model_hash' in event['details']:
            print(f"  {event['event_type']}: {event['details']['model_hash']}")
