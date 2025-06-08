import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import uuid
import warnings
warnings.filterwarnings('ignore')

class BehaviorPredictionDigitalThread:
    """
    ML-powered behavior prediction for cyber threat impact assessment
    """
    
    def __init__(self):
        self.thread_events = []
        self.models = {}
        self.predictions = []
        self.threat_scenarios = {}
        
    def log_event(self, event_type: str, details: dict, prediction_info: dict = None):
        """Log events in the digital thread"""
        event = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'prediction_info': prediction_info or {}
        }
        self.thread_events.append(event)
        return event['event_id']
    
    def generate_threat_scenario_data(self, n_scenarios=2000):
        """Generate synthetic threat scenario data with outcomes"""
        np.random.seed(42)
        
        # Threat characteristics
        threat_types = ['malware', 'phishing', 'ddos', 'insider_threat', 'apt']
        target_systems = ['web_server', 'database', 'workstation', 'network_device', 'cloud_service']
        attack_vectors = ['email', 'network', 'usb', 'web', 'social_engineering']
        
        scenarios = []
        for i in range(n_scenarios):
            # Basic threat features
            threat_type = np.random.choice(threat_types)
            target_system = np.random.choice(target_systems)
            attack_vector = np.random.choice(attack_vectors)
            
            # Threat sophistication (1-10)
            sophistication = np.random.randint(1, 11)
            
            # Organizational factors
            security_maturity = np.random.uniform(1, 10)  # 1-10 scale
            employee_training = np.random.uniform(1, 10)
            patch_level = np.random.uniform(1, 10)
            network_segmentation = np.random.uniform(1, 10)
            
            # Environmental factors
            business_hours = np.random.choice([0, 1])  # 0=off-hours, 1=business hours
            weekend = np.random.choice([0, 1])
            holiday = np.random.choice([0, 1], p=[0.9, 0.1])
            
            # Calculate impact based on realistic relationships
            base_impact = sophistication * 0.3
            
            # Threat type modifiers
            threat_multipliers = {
                'malware': 1.2, 'phishing': 0.8, 'ddos': 1.0, 
                'insider_threat': 1.5, 'apt': 2.0
            }
            base_impact *= threat_multipliers[threat_type]
            
            # Target system modifiers
            target_multipliers = {
                'web_server': 1.1, 'database': 1.8, 'workstation': 0.7,
                'network_device': 1.4, 'cloud_service': 1.3
            }
            base_impact *= target_multipliers[target_system]
            
            # Defense effectiveness
            defense_effectiveness = (security_maturity + employee_training + 
                                   patch_level + network_segmentation) / 4
            impact_reduction = defense_effectiveness * 0.15
            
            # Time-based factors
            if business_hours:
                base_impact *= 1.3  # Higher impact during business hours
            if weekend:
                base_impact *= 0.8  # Lower impact on weekends
            if holiday:
                base_impact *= 0.6  # Much lower impact on holidays
            
            # Final impact calculation
            financial_impact = max(0, base_impact - impact_reduction + np.random.normal(0, 0.5))
            
            # Operational impact (hours of downtime)
            operational_impact = financial_impact * np.random.uniform(0.5, 2.0)
            
            # Recovery time (hours)
            recovery_time = operational_impact * np.random.uniform(0.3, 1.5)
            
            # Success probability
            success_prob = min(0.95, max(0.05, 
                (sophistication * 0.08) - (defense_effectiveness * 0.06) + 
                np.random.normal(0, 0.1)
            ))
            
            scenario = {
                'threat_type': threat_type,
                'target_system': target_system,
                'attack_vector': attack_vector,
                'sophistication': sophistication,
                'security_maturity': security_maturity,
                'employee_training': employee_training,
                'patch_level': patch_level,
                'network_segmentation': network_segmentation,
                'business_hours': business_hours,
                'weekend': weekend,
                'holiday': holiday,
                'financial_impact': financial_impact,
                'operational_impact': operational_impact,
                'recovery_time': recovery_time,
                'success_probability': success_prob
            }
            scenarios.append(scenario)
        
        return pd.DataFrame(scenarios)
    
    def train_impact_prediction_models(self, data: pd.DataFrame):
        """Train ML models to predict threat impact with thread tracking"""
        
        # Log data preparation
        data_event_id = self.log_event(
            'threat_data_preparation',
            {
                'total_scenarios': len(data),
                'threat_types': data['threat_type'].unique().tolist(),
                'target_systems': data['target_system'].unique().tolist(),
                'avg_financial_impact': float(data['financial_impact'].mean()),
                'avg_success_probability': float(data['success_probability'].mean())
            }
        )
        
        # Encode categorical variables
        label_encoders = {}
        categorical_features = ['threat_type', 'target_system', 'attack_vector']
        
        for feature in categorical_features:
            le = LabelEncoder()
            data[f'{feature}_encoded'] = le.fit_transform(data[feature])
            label_encoders[feature] = le
        
        # Prepare features
        feature_columns = [
            'threat_type_encoded', 'target_system_encoded', 'attack_vector_encoded',
            'sophistication', 'security_maturity', 'employee_training', 
            'patch_level', 'network_segmentation', 'business_hours', 'weekend', 'holiday'
        ]
        
        X = data[feature_columns]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        scaling_event_id = self.log_event(
            'feature_preprocessing',
            {
                'categorical_encoding': 'label_encoding',
                'numerical_scaling': 'standard_scaling',
                'total_features': len(feature_columns)
            }
        )
        
        # Split data
        X_train, X_test, y_train_financial, y_test_financial = train_test_split(
            X_scaled, data['financial_impact'], test_size=0.2, random_state=42
        )
        
        _, _, y_train_success, y_test_success = train_test_split(
            X_scaled, (data['success_probability'] > 0.5).astype(int), 
            test_size=0.2, random_state=42
        )
        
        # Train financial impact regression model
        financial_model = RandomForestRegressor(n_estimators=100, random_state=42)
        financial_model.fit(X_train, y_train_financial)
        
        financial_predictions = financial_model.predict(X_test)
        financial_mse = mean_squared_error(y_test_financial, financial_predictions)
        
        financial_event_id = self.log_event(
            'financial_impact_model_training',
            {
                'algorithm': 'RandomForestRegressor',
                'n_estimators': 100,
                'training_samples': len(X_train),
                'test_mse': float(financial_mse),
                'feature_importance': dict(zip(feature_columns, financial_model.feature_importances_))
            },
            prediction_info={
                'target_variable': 'financial_impact',
                'model_type': 'regression'
            }
        )
        
        # Train success probability classification model
        success_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        success_model.fit(X_train, y_train_success)
        
        success_predictions = success_model.predict(X_test)
        success_accuracy = accuracy_score(y_test_success, success_predictions)
        
        success_event_id = self.log_event(
            'success_probability_model_training',
            {
                'algorithm': 'GradientBoostingClassifier',
                'n_estimators': 100,
                'training_samples': len(X_train),
                'test_accuracy': float(success_accuracy),
                'feature_importance': dict(zip(feature_columns, success_model.feature_importances_))
            },
            prediction_info={
                'target_variable': 'success_probability',
                'model_type': 'classification'
            }
        )
        
        # Store models and preprocessing objects
        self.models['behavior_prediction'] = {
            'financial_model': financial_model,
            'success_model': success_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_columns': feature_columns,
            'training_event_ids': [financial_event_id, success_event_id]
        }
        
        return financial_model, success_model, scaler, label_encoders
    
    def predict_threat_behavior(self, threat_scenarios: list):
        """Predict behavior and impact of new threat scenarios"""
        
        if 'behavior_prediction' not in self.models:
            raise ValueError("Models not trained yet!")
        
        model_info = self.models['behavior_prediction']
        financial_model = model_info['financial_model']
        success_model = model_info['success_model']
        scaler = model_info['scaler']
        label_encoders = model_info['label_encoders']
        feature_columns = model_info['feature_columns']
        
        predictions = []
        
        for scenario in threat_scenarios:
            # Encode categorical features
            scenario_encoded = scenario.copy()
            for feature, encoder in label_encoders.items():
                if scenario[feature] in encoder.classes_:
                    scenario_encoded[f'{feature}_encoded'] = encoder.transform([scenario[feature]])[0]
                else:
                    # Handle unseen categories
                    scenario_encoded[f'{feature}_encoded'] = 0
            
            # Prepare feature vector
            feature_vector = np.array([[scenario_encoded[col] for col in feature_columns]])
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Make predictions
            financial_impact_pred = financial_model.predict(feature_vector_scaled)[0]
            success_prob_pred = success_model.predict_proba(feature_vector_scaled)[0][1]
            
            # Calculate derived metrics
            expected_impact = financial_impact_pred * success_prob_pred
            risk_level = self._calculate_risk_level(financial_impact_pred, success_prob_pred)
            
            prediction_result = {
                'scenario_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'input_scenario': scenario,
                'predicted_financial_impact': float(financial_impact_pred),
                'predicted_success_probability': float(success_prob_pred),
                'expected_impact': float(expected_impact),
                'risk_level': risk_level,
                'confidence_score': self._calculate_confidence(financial_model, success_model, feature_vector_scaled)
            }
            
            predictions.append(prediction_result)
            
            # Log individual prediction
            self.log_event(
                'threat_behavior_prediction',
                {
                    'threat_type': scenario['threat_type'],
                    'target_system': scenario['target_system'],
                    'predicted_financial_impact': float(financial_impact_pred),
                    'predicted_success_probability': float(success_prob_pred),
                    'risk_level': risk_level
                },
                prediction_info={
                    'models_used': ['financial_impact_model', 'success_probability_model'],
                    'confidence_score': prediction_result['confidence_score']
                }
            )
        
        # Log batch prediction summary
        batch_event_id = self.log_event(
            'batch_threat_prediction',
            {
                'scenarios_processed': len(threat_scenarios),
                'avg_predicted_impact': float(np.mean([p['predicted_financial_impact'] for p in predictions])),
                'avg_success_probability': float(np.mean([p['predicted_success_probability'] for p in predictions])),
                'high_risk_scenarios': len([p for p in predictions if p['risk_level'] == 'high'])
            }
        )
        
        self.predictions.extend(predictions)
        return predictions
    
    def _calculate_risk_level(self, financial_impact: float, success_prob: float) -> str:
        """Calculate risk level based on impact and probability"""
        risk_score = financial_impact * success_prob
        
        if risk_score > 3.0:
            return 'critical'
        elif risk_score > 2.0:
            return 'high'
        elif risk_score > 1.0:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence(self, financial_model, success_model, feature_vector):
        """Calculate prediction confidence based on model uncertainty"""
        # For Random Forest, use prediction variance across trees
        financial_predictions = np.array([tree.predict(feature_vector)[0] for tree in financial_model.estimators_])
        financial_std = np.std(financial_predictions)
        
        # For Gradient Boosting, use prediction probability
        success_proba = success_model.predict_proba(feature_vector)[0]
        success_confidence = max(success_proba)  # Confidence is max probability
        
        # Combine confidences (inverse of uncertainty)
        financial_confidence = 1 / (1 + financial_std)
        overall_confidence = (financial_confidence + success_confidence) / 2
        
        return float(overall_confidence)
    
    def simulate_threat_scenarios(self):
        """Simulate various threat scenarios for prediction"""
        scenarios = [
            {
                'threat_type': 'apt',
                'target_system': 'database',
                'attack_vector': 'email',
                'sophistication': 9,
                'security_maturity': 6.0,
                'employee_training': 5.0,
                'patch_level': 7.0,
                'network_segmentation': 8.0,
                'business_hours': 1,
                'weekend': 0,
                'holiday': 0
            },
            {
                'threat_type': 'ddos',
                'target_system': 'web_server',
                'attack_vector': 'network',
                'sophistication': 6,
                'security_maturity': 8.0,
                'employee_training': 7.0,
                'patch_level': 9.0,
                'network_segmentation': 6.0,
                'business_hours': 1,
                'weekend': 0,
                'holiday': 0
            },
            {
                'threat_type': 'phishing',
                'target_system': 'workstation',
                'attack_vector': 'email',
                'sophistication': 4,
                'security_maturity': 4.0,
                'employee_training': 3.0,
                'patch_level': 5.0,
                'network_segmentation': 4.0,
                'business_hours': 1,
                'weekend': 0,
                'holiday': 0
            },
            {
                'threat_type': 'insider_threat',
                'target_system': 'cloud_service',
                'attack_vector': 'social_engineering',
                'sophistication': 7,
                'security_maturity': 7.0,
                'employee_training': 6.0,
                'patch_level': 8.0,
                'network_segmentation': 5.0,
                'business_hours': 0,
                'weekend': 1,
                'holiday': 0
            }
        ]
        
        return scenarios
    
    def visualize_predictions(self, predictions: list):
        """Visualize threat behavior predictions"""
        if not predictions:
            print("No predictions to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Risk level distribution
        risk_levels = [p['risk_level'] for p in predictions]
        risk_counts = pd.Series(risk_levels).value_counts()
        axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Risk Level Distribution')
        
        # Financial impact vs Success probability
        financial_impacts = [p['predicted_financial_impact'] for p in predictions]
        success_probs = [p['predicted_success_probability'] for p in predictions]
        scatter = axes[0, 1].scatter(success_probs, financial_impacts, 
                                   c=[hash(p['risk_level']) for p in predictions], 
                                   cmap='viridis', alpha=0.7)
        axes[0, 1].set_xlabel('Predicted Success Probability')
        axes[0, 1].set_ylabel('Predicted Financial Impact')
        axes[0, 1].set_title('Impact vs Success Probability')
        
        # Threat type analysis
        threat_data = []
        for p in predictions:
            threat_data.append({
                'threat_type': p['input_scenario']['threat_type'],
                'expected_impact': p['expected_impact']
            })
        
        threat_df = pd.DataFrame(threat_data)
        threat_summary = threat_df.groupby('threat_type')['expected_impact'].mean().sort_values(ascending=True)
        axes[1, 0].barh(threat_summary.index, threat_summary.values)
        axes[1, 0].set_xlabel('Average Expected Impact')
        axes[1, 0].set_title('Expected Impact by Threat Type')
        
        # Confidence scores
        confidence_scores = [p['confidence_score'] for p in predictions]
        axes[1, 1].hist(confidence_scores, bins=10, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Confidence Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def generate_prediction_report(self, filename: str = "behavior_prediction_report.json"):
        """Generate comprehensive behavior prediction report"""
        report = {
            'report_generated': datetime.now().isoformat(),
            'summary': {
                'total_predictions': len(self.predictions),
                'avg_financial_impact': float(np.mean([p['predicted_financial_impact'] for p in self.predictions])) if self.predictions else 0,
                'avg_success_probability': float(np.mean([p['predicted_success_probability'] for p in self.predictions])) if self.predictions else 0,
                'risk_distribution': self._get_risk_distribution()
            },
            'digital_thread': self.thread_events,
            'predictions': self.predictions,
            'model_lineage': self._get_model_lineage()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Behavior prediction report exported to {filename}")
        return filename
    
    def _get_risk_distribution(self):
        """Get distribution of risk levels"""
        if not self.predictions:
            return {}
        
        risk_levels = [p['risk_level'] for p in self.predictions]
        return dict(pd.Series(risk_levels).value_counts())
    
    def _get_model_lineage(self):
        """Get model training and prediction lineage"""
        lineage = []
        for event in self.thread_events:
            if event['event_type'] in ['financial_impact_model_training', 'success_probability_model_training', 
                                     'threat_behavior_prediction', 'batch_threat_prediction']:
                lineage.append({
                    'timestamp': event['timestamp'],
                    'event_type': event['event_type'],
                    'details': event['details']
                })
        return lineage

# Demo execution
if __name__ == "__main__":
    print("=== ML-Powered Behavior Prediction Digital Thread Demo ===\n")
    
    # Initialize behavior prediction system
    behavior_system = BehaviorPredictionDigitalThread()
    
    # Generate training data
    print("Step 1: Generating threat scenario training data...")
    training_data = behavior_system.generate_threat_scenario_data(n_scenarios=2000)
    
    # Train models
    print("Step 2: Training behavior prediction models...")
    financial_model, success_model, scaler, encoders = behavior_system.train_impact_prediction_models(training_data)
    
    # Simulate new threat scenarios
    print("Step 3: Simulating new threat scenarios...")
    new_scenarios = behavior_system.simulate_threat_scenarios()
    
    # Make predictions
    print("Step 4: Predicting threat behavior and impact...")
    predictions = behavior_system.predict_threat_behavior(new_scenarios)
    
    # Display results
    print(f"\n=== Behavior Prediction Results ===")
    for i, prediction in enumerate(predictions):
        scenario = prediction['input_scenario']
        print(f"\nScenario {i+1}: {scenario['threat_type'].upper()} -> {scenario['target_system']}")
        print(f"  Predicted Financial Impact: ${prediction['predicted_financial_impact']:.2f}K")
        print(f"  Success Probability: {prediction['predicted_success_probability']:.1%}")
        print(f"  Expected Impact: ${prediction['expected_impact']:.2f}K")
        print(f"  Risk Level: {prediction['risk_level'].upper()}")
        print(f"  Confidence: {prediction['confidence_score']:.2f}")
    
    # Show summary statistics
    print(f"\n=== Prediction Summary ===")
    avg_impact = np.mean([p['predicted_financial_impact'] for p in predictions])
    avg_success = np.mean([p['predicted_success_probability'] for p in predictions])
    risk_dist = behavior_system._get_risk_distribution()
    
    print(f"Average Predicted Impact: ${avg_impact:.2f}K")
    print(f"Average Success Probability: {avg_success:.1%}")
    print(f"Risk Distribution: {risk_dist}")
    
    # Visualize predictions
    behavior_system.visualize_predictions(predictions)
    
    # Generate report
    report_file = behavior_system.generate_prediction_report()
    
    print(f"\n=== Digital Thread Summary ===")
    print(f"Total thread events: {len(behavior_system.thread_events)}")
    print(f"Models trained: Financial Impact, Success Probability")
    print(f"Predictions made: {len(behavior_system.predictions)}")
