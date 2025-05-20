import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

def generate_multi_source_data(n_samples=1000, n_features=20):
    """Generate synthetic data from multiple sources"""
    # Source 1: Main classification data
    X1, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=10, n_redundant=5, 
                               random_state=42)
    
    # Source 2: Additional features with some correlation to target
    X2 = np.random.randn(n_samples, 15)
    X2[:, :5] = X2[:, :5] + 0.3 * y.reshape(-1, 1)
    
    # Source 3: Weak signal source
    X3 = np.random.randn(n_samples, 10)
    X3[:, :3] = X3[:, :3] + 0.1 * y.reshape(-1, 1)
    
    return X1, X2, X3, y

def feature_based_fusion():
    """
    Feature-based fusion: Combines raw features from multiple sources
    before classification.
    """
    print("\n=== FEATURE-BASED FUSION ===")
    
    # Generate data from three sources
    X1, X2, X3, y = generate_multi_source_data()
    
    # Split into training and testing sets
    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(
        X1, X2, X3, y, test_size=0.3, random_state=42)
    
    # Individual source performance
    print("Individual source performance:")
    for i, (X_train, X_test) in enumerate([(X1_train, X1_test), 
                                          (X2_train, X2_test), 
                                          (X3_train, X3_test)]):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"  Source {i+1} accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Feature-level fusion
    # Concatenate features from all sources
    X_train_fused = np.hstack((X1_train, X2_train, X3_train))
    X_test_fused = np.hstack((X1_test, X2_test, X3_test))
    
    # Normalize the fused features
    scaler = StandardScaler()
    X_train_fused = scaler.fit_transform(X_train_fused)
    X_test_fused = scaler.transform(X_test_fused)
    
    # Train classifier on fused features
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_fused, y_train)
    y_pred = clf.predict(X_test_fused)
    
    print(f"Feature fusion accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Feature importance analysis
    feature_importances = clf.feature_importances_
    source_importances = [
        np.sum(feature_importances[:X1.shape[1]]),
        np.sum(feature_importances[X1.shape[1]:X1.shape[1]+X2.shape[1]]),
        np.sum(feature_importances[X1.shape[1]+X2.shape[1]:])
    ]
    
    print("Source contribution to feature fusion:")
    for i, importance in enumerate(source_importances):
        print(f"  Source {i+1}: {importance:.4f}")

def score_based_fusion():
    """
    Score-based fusion: Combines probability scores from multiple classifiers
    trained on different data sources.
    """
    print("\n=== SCORE-BASED FUSION ===")
    
    # Generate data from three sources
    X1, X2, X3, y = generate_multi_source_data()
    
    # Split into training and testing sets
    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(
        X1, X2, X3, y, test_size=0.3, random_state=42)
    
    # Train a classifier on each source
    classifiers = []
    for i, (X_train, X_test) in enumerate([(X1_train, X1_test), 
                                          (X2_train, X2_test), 
                                          (X3_train, X3_test)]):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        classifiers.append(clf)
    
    # Get probability scores from each classifier
    proba_1 = classifiers[0].predict_proba(X1_test)
    proba_2 = classifiers[1].predict_proba(X2_test)
    proba_3 = classifiers[2].predict_proba(X3_test)
    
    # Different score fusion methods
    
    # 1. Simple averaging of probabilities
    avg_proba = (proba_1 + proba_2 + proba_3) / 3
    avg_pred = np.argmax(avg_proba, axis=1)
    print(f"Score fusion (average) accuracy: {accuracy_score(y_test, avg_pred):.4f}")
    
    # 2. Weighted averaging (based on individual performance)
    # First, determine weights based on validation performance
    weights = []
    for i, (clf, X_val, src_name) in enumerate(zip(classifiers, 
                                                 [X1_test, X2_test, X3_test],
                                                 ['Source 1', 'Source 2', 'Source 3'])):
        val_pred = clf.predict(X_val)
        acc = accuracy_score(y_test, val_pred)
        weights.append(acc)
        print(f"  {src_name} accuracy: {acc:.4f}")
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    print(f"  Fusion weights: {weights}")
    
    # Apply weighted average
    weighted_proba = (weights[0] * proba_1 + 
                      weights[1] * proba_2 + 
                      weights[2] * proba_3)
    weighted_pred = np.argmax(weighted_proba, axis=1)
    print(f"Score fusion (weighted) accuracy: {accuracy_score(y_test, weighted_pred):.4f}")
    
    # 3. Product rule (multiply probabilities)
    product_proba = proba_1 * proba_2 * proba_3
    product_pred = np.argmax(product_proba, axis=1)
    print(f"Score fusion (product) accuracy: {accuracy_score(y_test, product_pred):.4f}")

def voting_based_fusion():
    """
    Voting-based fusion: Combines decisions from multiple classifiers
    through majority voting or similar methods.
    """
    print("\n=== VOTING-BASED FUSION ===")
    
    # Generate data from three sources
    X1, X2, X3, y = generate_multi_source_data()
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X1, y, test_size=0.3, random_state=42)
    
    # Create different classifiers
    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf3 = SVC(probability=True, random_state=42)
    
    # Train individual classifiers
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)
    
    # Evaluate individual classifiers
    for i, clf in enumerate([clf1, clf2, clf3]):
        y_pred = clf.predict(X_test)
        print(f"Classifier {i+1} accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # 1. Hard voting (majority vote)
    hard_voting = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
        voting='hard'
    )
    hard_voting.fit(X_train, y_train)
    hard_pred = hard_voting.predict(X_test)
    print(f"Hard voting accuracy: {accuracy_score(y_test, hard_pred):.4f}")
    
    # 2. Soft voting (weighted probabilities)
    soft_voting = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
        voting='soft'
    )
    soft_voting.fit(X_train, y_train)
    soft_pred = soft_voting.predict(X_test)
    print(f"Soft voting accuracy: {accuracy_score(y_test, soft_pred):.4f}")
    
    # 3. Custom voting (manual implementation)
    # Get predictions from each classifier
    pred1 = clf1.predict(X_test)
    pred2 = clf2.predict(X_test)
    pred3 = clf3.predict(X_test)
    
    # Stack predictions
    all_preds = np.vstack((pred1, pred2, pred3)).T
    
    # Majority vote
    from scipy.stats import mode
    custom_pred = mode(all_preds, axis=1)[0].flatten()
    print(f"Custom majority vote accuracy: {accuracy_score(y_test, custom_pred):.4f}")

def main():
    print("DATA FUSION EXAMPLES")
    print("====================")
    
    # Run all fusion examples
    feature_based_fusion()
    score_based_fusion()
    voting_based_fusion()
    
    print("\nSUMMARY OF DATA FUSION APPROACHES:")
    print("1. Feature-based fusion: Combines raw data from multiple sources before classification")
    print("2. Score-based fusion: Combines probability scores from multiple classifiers")
    print("3. Voting-based fusion: Combines decisions from multiple classifiers")

if __name__ == "__main__":
    main()