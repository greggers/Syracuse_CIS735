import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import uuid
import time
from threading import Thread
import queue

class DynamicDefenseDigitalThread:
    """
    ML-powered dynamic defense system with adaptive security measures
    """
    
    def __init__(self):
        self.thread_events = []
        self.models = {}
        self.defense_actions = []
        self.threat_queue = queue.Queue()
        self.defense_policies = {}
        self.system_state = {
            'threat_level': 'low',
            'active_defenses': [],
            'blocked_ips': set(),
            'quarantined_files': set(),
            'network_restrictions': {}
        }
        
    def log_event(self, event_type: str, details: dict, defense_info: dict = None):
        """Log events in the digital thread"""
        event = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'defense_info': defense_info or {},
            'system_state_snapshot': self.system_state.copy()
        }
        self.thread_events.append(event)
        return event['event_id']
    
    def generate_real_time_threats(self, duration_minutes=5):
        """Generate real-time threat data stream"""
        threat_types = ['malware', 'phishing', 'ddos', 'brute_force', 'data_exfiltration']
        severity_levels = ['low', 'medium', 'high', 'critical']
        
        def threat_generator():
            start_time = datetime.now()
            threat_id = 0
            
            while (datetime.now() - start_time).seconds < duration_minutes * 60:
                # Generate threat with varying frequency based on time
                base_interval = np.random.exponential(2.0)  # Average 2 seconds between threats
                
                # Simulate threat bursts (higher frequency during attacks)
                if np.random.random() < 0.1:  # 10% chance of burst
                    base_interval *= 0.1  # Much faster during bursts
                
                time.sleep(base_interval)
                
                threat_id += 1
                threat = {
                    'threat_id': f'T{threat_id:04d}',
                    'timestamp': datetime.now().isoformat(),
                    'threat_type': np.random.choice(threat_types),
                    'severity': np.random.choice(severity_levels, p=[0.4, 0.3, 0.2, 0.1]),
                    'source_ip': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                    'target_port': np.random.choice([80, 443, 22, 21, 25, 53, 3389]),
                    'payload_size': np.random.exponential(1000),
                    'connection_count': np.random.poisson(5),
                    'is_encrypted': np.random.choice([0, 1], p=[0.3, 0.7]),
                    'geolocation_risk': np.random.uniform(0, 1),
                    'reputation_score': np.random.uniform(0, 1)
                }
                
                self.threat_queue.put(threat)
        
        # Start threat generation in background thread
        threat_thread = Thread(target=threat_generator)
        threat_thread.daemon = True
        threat_thread.start()
        
        return threat_thread
    
    def train_threat_classification_model(self, historical_data: pd.DataFrame):
        """Train ML model for real-time threat classification"""
        
        # Log training initiation
        training_event_id = self.log_event(
            'threat_classification_training_start',
            {
                'training_samples': len(historical_data),
                'threat_types': historical_data['threat_type'].unique().tolist(),
                'severity_distribution': historical_data['severity'].value_counts().to_dict()
            }
        )
        
        # Prepare features
        feature_columns = [
            'target_port', 'payload_size', 'connection_count', 
            'is_encrypted', 'geolocation_risk', 'reputation_score'
        ]
        
        X = historical_data[feature_columns]
        y = historical_data['severity']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_scaled, y)
        
        # Calculate feature importance
        feature_importance = dict(zip(feature_columns, classifier.feature_importances_))
        
        # Store model
        self.models['threat_classifier'] = {
            'model': classifier,
            'scaler': scaler,
            'features': feature_columns,
            'classes': classifier.classes_.tolist()
        }
        
        # Log training completion
        self.log_event(
            'threat_classification_training_complete',
            {
                'model_accuracy': float(classifier.score(X_scaled, y)),
                'feature_importance': feature_importance,
                'model_classes': classifier.classes_.tolist()
            },
            defense_info={
                'model_type': 'threat_classification',
                'algorithm': 'RandomForestClassifier'
            }
        )
        
        return classifier, scaler
    
    def initialize_defense_policies(self):
        """Initialize adaptive defense policies"""
        self.defense_policies = {
            'low': {
                'actions': ['log_event'],
                'thresholds': {'block_threshold': 0.9, 'quarantine_threshold': 0.95},
                'response_time': 30  # seconds
            },
            'medium': {
                'actions': ['log_event', 'rate_limit', 'enhanced_monitoring'],
                'thresholds': {'block_threshold': 0.7, 'quarantine_threshold': 0.8},
                'response_time': 15
            },
            'high': {
                'actions': ['log_event', 'rate_limit', 'enhanced_monitoring', 'ip_blocking'],
                'thresholds': {'block_threshold': 0.5, 'quarantine_threshold': 0.6},
                'response_time': 5
            },
            'critical': {
                'actions': ['log_event', 'rate_limit', 'enhanced_monitoring', 'ip_blocking', 
                           'network_isolation', 'emergency_response'],
                'thresholds': {'block_threshold': 0.3, 'quarantine_threshold': 0.4},
                'response_time': 1
            }
        }
        
        self.log_event(
            'defense_policies_initialized',
            {
                'policy_levels': list(self.defense_policies.keys()),
                'total_defense_actions': len(set().union(*[p['actions'] for p in self.defense_policies.values()]))
            }
        )
    
    def classify_threat_severity(self, threat: dict):
        """Classify threat severity using ML model"""
        if 'threat_classifier' not in self.models:
            # Fallback to rule-based classification
            return self._rule_based_classification(threat)
        
        model_info = self.models['threat_classifier']
        classifier = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features']
        
        # Prepare feature vector
        feature_vector = np.array([[threat[feature] for feature in features]])
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Predict severity and confidence
        predicted_severity = classifier.predict(feature_vector_scaled)[0]
        prediction_proba = classifier.predict_proba(feature_vector_scaled)[0]
        confidence = np.max(prediction_proba)
        
        return predicted_severity, confidence
    
    def _rule_based_classification(self, threat: dict):
        """Fallback rule-based threat classification"""
        score = 0
        
        # Threat type scoring
        threat_scores = {
            'ddos': 0.8, 'malware': 0.9, 'data_exfiltration': 0.85,
            'brute_force': 0.6, 'phishing': 0.5
        }
        score += threat_scores.get(threat['threat_type'], 0.3)
        
        # Additional factors
        if threat['payload_size'] > 10000:
            score += 0.2
        if threat['connection_count'] > 10:
            score += 0.15
        if threat['geolocation_risk'] > 0.7:
            score += 0.1
        if threat['reputation_score'] < 0.3:
            score += 0.2
        
        # Map score to severity
        if score >= 0.8:
            return 'critical', 0.8
        elif score >= 0.6:
            return 'high', 0.7
        elif score >= 0.4:
            return 'medium', 0.6
        else:
            return 'low', 0.5
    
    def execute_defense_actions(self, threat: dict, severity: str, confidence: float):
        """Execute adaptive defense actions based on threat assessment"""
        
        if severity not in self.defense_policies:
            severity = 'low'
        
        policy = self.defense_policies[severity]
        actions_taken = []
        
        for action in policy['actions']:
            success = self._execute_single_action(action, threat, severity, confidence)
            if success:
                actions_taken.append(action)
        
        # Update system threat level
        self._update_system_threat_level(severity)
        
        # Log defense actions
        defense_event_id = self.log_event(
            'defense_actions_executed',
            {
                'threat_id': threat['threat_id'],
                'threat_type': threat['threat_type'],
                'assessed_severity': severity,
                'confidence': float(confidence),
                'actions_taken': actions_taken,
                'response_time': policy['response_time']
            },
            defense_info={
                'policy_applied': severity,
                'total_actions': len(actions_taken),
                'system_threat_level': self.system_state['threat_level']
            }
        )
        
        defense_action = {
            'timestamp': datetime.now().isoformat(),
            'threat_id': threat['threat_id'],
            'severity': severity,
            'confidence': confidence,
            'actions_taken': actions_taken,
            'event_id': defense_event_id
        }
        
        self.defense_actions.append(defense_action)
        return defense_action
    
    def _execute_single_action(self, action: str, threat: dict, severity: str, confidence: float):
        """Execute a single defense action"""
        try:
            if action == 'log_event':
                return True  # Always successful
            
            elif action == 'rate_limit':
                # Simulate rate limiting
                source_ip = threat['source_ip']
                if source_ip not in self.system_state['network_restrictions']:
                    self.system_state['network_restrictions'][source_ip] = {
                        'rate_limit': True,
                        'max_connections': 10,
                        'timestamp': datetime.now().isoformat()
                    }
                return True
            
            elif action == 'enhanced_monitoring':
                # Enable enhanced monitoring
                if 'enhanced_monitoring' not in self.system_state['active_defenses']:
                    self.system_state['active_defenses'].append('enhanced_monitoring')
                return True
            
            elif action == 'ip_blocking':
                # Block suspicious IP
                source_ip = threat['source_ip']
                if confidence > 0.7:  # Only block with high confidence
                    self.system_state['blocked_ips'].add(source_ip)
                    return True
                return False
            
            elif action == 'network_isolation':
                # Isolate network segment
                if 'network_isolation' not in self.system_state['active_defenses']:
                    self.system_state['active_defenses'].append('network_isolation')
                return True
            
            elif action == 'emergency_response':
                # Trigger emergency response
                if 'emergency_response' not in self.system_state['active_defenses']:
                    self.system_state['active_defenses'].append('emergency_response')
                return True
            
            else:
                return False
                
        except Exception as e:
            return False
    
    def _update_system_threat_level(self, new_severity: str):
        """Update overall system threat level based on recent threats"""
        severity_weights = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        current_weight = severity_weights.get(self.system_state['threat_level'], 1)
        new_weight = severity_weights.get(new_severity, 1)
        
        # Weighted average with recent threats having more influence
        updated_weight = (current_weight * 0.7) + (new_weight * 0.3)
        
        if updated_weight >= 3.5:
            self.system_state['threat_level'] = 'critical'
        elif updated_weight >= 2.5:
            self.system_state['threat_level'] = 'high'
        elif updated_weight >= 1.5:
            self.system_state['threat_level'] = 'medium'
        else:
            self.system_state['threat_level'] = 'low'
    
    def process_real_time_threats(self, duration_minutes=2):
        """Process threats in real-time with adaptive defense"""
        
        print(f"Starting real-time threat processing for {duration_minutes} minutes...")
        
        # Generate historical data for training
        historical_threats = self._generate_historical_data(1000)
        
        # Train classification model
        self.train_threat_classification_model(historical_threats)
        
        # Initialize defense policies
        self.initialize_defense_policies()
        
        # Start threat generation
        threat_thread = self.generate_real_time_threats(duration_minutes)
        
        # Process threats in real-time
        start_time = datetime.now()
        processed_threats = 0
        
        while (datetime.now() - start_time).seconds < duration_minutes * 60:
            try:
                # Get threat from queue (timeout to check for end condition)
                threat = self.threat_queue.get(timeout=1)
                
                # Classify threat severity
                severity, confidence = self.classify_threat_severity(threat)
                
                # Execute adaptive defense
                defense_action = self.execute_defense_actions(threat, severity, confidence)
                
                processed_threats += 1
                
                # Print real-time updates
                if processed_threats % 10 == 0:
                    print(f"Processed {processed_threats} threats. Current threat level: {self.system_state['threat_level']}")
                
            except queue.Empty:
                continue  # No threats in queue, continue monitoring
            except Exception as e:
                print(f"Error processing threat: {e}")
                continue
        
        print(f"Real-time processing complete. Total threats processed: {processed_threats}")
        return processed_threats
    
    def _generate_historical_data(self, n_samples: int):
        """Generate historical threat data for model training"""
        np.random.seed(42)
        
        threat_types = ['malware', 'phishing', 'ddos', 'brute_force', 'data_exfiltration']
        severity_levels = ['low', 'medium', 'high', 'critical']
        
        data = []
        for i in range(n_samples):
            threat_type = np.random.choice(threat_types)
            
            # Generate features with realistic correlations
            if threat_type == 'ddos':
                payload_size = np.random.exponential(5000)
                connection_count = np.random.poisson(20)
                severity = np.random.choice(['high', 'critical'], p=[0.6, 0.4])
            elif threat_type == 'malware':
                payload_size = np.random.exponential(2000)
                connection_count = np.random.poisson(3)
                severity = np.random.choice(['medium', 'high', 'critical'], p=[0.3, 0.5, 0.2])
            else:
                payload_size = np.random.exponential(1000)
                connection_count = np.random.poisson(5)
                severity = np.random.choice(severity_levels, p=[0.4, 0.3, 0.2, 0.1])
            
            threat = {
                'threat_type': threat_type,
                'severity': severity,
                'target_port': np.random.choice([80, 443, 22, 21, 25, 53, 3389]),
                'payload_size': payload_size,
                'connection_count': connection_count,
                'is_encrypted': np.random.choice([0, 1], p=[0.3, 0.7]),
                'geolocation_risk': np.random.uniform(0, 1),
                'reputation_score': np.random.uniform(0, 1)
            }
            data.append(threat)
        
        return pd.DataFrame(data)
    
    def analyze_defense_effectiveness(self):
        """Analyze the effectiveness of dynamic defense actions"""
        if not self.defense_actions:
            print("No defense actions to analyze")
            return {}
        
        # Calculate defense metrics
        total_actions = len(self.defense_actions)
        severity_distribution = {}
        action_effectiveness = {}
        
        for action in self.defense_actions:
            severity = action['severity']
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
            
            for taken_action in action['actions_taken']:
                if taken_action not in action_effectiveness:
                    action_effectiveness[taken_action] = {'count': 0, 'avg_confidence': 0}
                action_effectiveness[taken_action]['count'] += 1
                action_effectiveness[taken_action]['avg_confidence'] += action['confidence']
        
        # Calculate average confidence for each action type
        for action_type in action_effectiveness:
            count = action_effectiveness[action_type]['count']
            action_effectiveness[action_type]['avg_confidence'] /= count
        
        analysis = {
            'total_defense_actions': total_actions,
            'severity_distribution': severity_distribution,
            'action_effectiveness': action_effectiveness,
            'blocked_ips_count': len(self.system_state['blocked_ips']),
            'active_defenses': len(self.system_state['active_defenses']),
            'current_threat_level': self.system_state['threat_level']
        }
        
        # Log analysis
        self.log_event(
            'defense_effectiveness_analysis',
            analysis,
            defense_info={
                'analysis_type': 'post_processing',
                'metrics_calculated': list(analysis.keys())
            }
        )
        
        return analysis
    
    def visualize_dynamic_defense(self):
        """Visualize dynamic defense system performance"""
        if not self.defense_actions:
            print("No defense actions to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Defense actions over time
        timestamps = [datetime.fromisoformat(action['timestamp']) for action in self.defense_actions]
        severities = [action['severity'] for action in self.defense_actions]
        
        severity_colors = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'}
        colors = [severity_colors.get(s, 'blue') for s in severities]
        
        axes[0, 0].scatter(timestamps, range(len(timestamps)), c=colors, alpha=0.7)
        axes[0, 0].set_title('Defense Actions Timeline by Severity')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Action Sequence')
        
        # Severity distribution
        severity_counts = pd.Series(severities).value_counts()
        axes[0, 1].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Threat Severity Distribution')
        
        # Action types frequency
        all_actions = []
        for action in self.defense_actions:
            all_actions.extend(action['actions_taken'])
        
        action_counts = pd.Series(all_actions).value_counts()
        axes[1, 0].bar(action_counts.index, action_counts.values)
        axes[1, 0].set_title('Defense Action Types Frequency')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Confidence scores distribution
        confidences = [action['confidence'] for action in self.defense_actions]
        axes[1, 1].hist(confidences, bins=20, alpha=0.7, color='blue')
        axes[1, 1].set_title('Threat Classification Confidence')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def generate_defense_report(self, filename: str = "dynamic_defense_report.json"):
        """Generate comprehensive dynamic defense report"""
        analysis = self.analyze_defense_effectiveness()
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'summary': {
                'total_defense_actions': len(self.defense_actions),
                'total_thread_events': len(self.thread_events),
                'current_system_state': self.system_state,
                'defense_effectiveness': analysis
            },
            'digital_thread': self.thread_events,
            'defense_actions': self.defense_actions,
            'defense_policies': self.defense_policies,
            'model_info': self._get_model_info()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Dynamic defense report exported to {filename}")
        return filename
    
    def _get_model_info(self):
        """Get information about trained models"""
        model_info = {}
        for model_name, model_data in self.models.items():
            model_info[model_name] = {
                'features': model_data.get('features', []),
                'classes': model_data.get('classes', []),
                'model_type': str(type(model_data['model']).__name__)
            }
        return model_info

# Demo execution
if __name__ == "__main__":
    print("=== ML-Powered Dynamic Defense Digital Thread Demo ===\n")
    
    # Initialize dynamic defense system
    defense_system = DynamicDefenseDigitalThread()
    
    # Process real-time threats with adaptive defense
    print("Step 1: Starting real-time threat processing with adaptive defense...")
    processed_count = defense_system.process_real_time_threats(duration_minutes=1)  # Short demo
    
    # Analyze defense effectiveness
    print("\nStep 2: Analyzing defense effectiveness...")
    analysis = defense_system.analyze_defense_effectiveness()
    
    # Display results
    print(f"\n=== Dynamic Defense Results ===")
    print(f"Threats processed: {processed_count}")
    print(f"Defense actions taken: {analysis['total_defense_actions']}")
    print(f"Current threat level: {analysis['current_threat_level']}")
    print(f"Blocked IPs: {analysis['blocked_ips_count']}")
    print(f"Active defenses: {analysis['active_defenses']}")
    
    print(f"\n=== Severity Distribution ===")
    for severity, count in analysis['severity_distribution'].items():
        print(f"  {severity}: {count}")
    
    print(f"\n=== Action Effectiveness ===")
    for action, stats in analysis['action_effectiveness'].items():
        print(f"  {action}: {stats['count']} times (avg confidence: {stats['avg_confidence']:.2f})")
    
    print(f"\n=== System State ===")
    print(f"Blocked IPs: {list(defense_system.system_state['blocked_ips'])}")
    print(f"Active Defenses: {defense_system.system_state['active_defenses']}")
    print(f"Network Restrictions: {len(defense_system.system_state['network_restrictions'])} IPs")
    
    # Visualize results
    defense_system.visualize_dynamic_defense()
    
    # Generate comprehensive report
    report_file = defense_system.generate_defense_report()
    
    print(f"\n=== Digital Thread Summary ===")
    print(f"Total thread events: {len(defense_system.thread_events)}")
    print(f"Defense policies: {list(defense_system.defense_policies.keys())}")
    print(f"Models trained: {list(defense_system.models.keys())}")
    
    # Show sample of recent thread events
    print(f"\n=== Recent Thread Events (Last 5) ===")
    for event in defense_system.thread_events[-5:]:
        print(f"  {event['timestamp']}: {event['event_type']}")

