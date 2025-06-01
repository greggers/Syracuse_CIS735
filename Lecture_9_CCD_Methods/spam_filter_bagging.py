import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

class SpamFilterBagging:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', lowercase=True)
        self.bagging_tree = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=10),
            n_estimators=50,
            random_state=42
        )
        self.bagging_nb = BaggingClassifier(
            base_estimator=MultinomialNB(),
            n_estimators=50,
            random_state=42
        )
        
    def create_sample_data(self):
        """
        Create sample email data with spam using concealment/deception tactics
        """
        # Legitimate emails
        legit_emails = [
            "Meeting scheduled for tomorrow at 3 PM in conference room A",
            "Please review the quarterly report and send feedback by Friday",
            "Your order has been shipped and will arrive within 3-5 business days",
            "Thank you for your presentation today, it was very informative",
            "Reminder: Annual performance reviews are due next week",
            "The project deadline has been extended to next month",
            "Welcome to our team! Please complete your onboarding tasks",
            "Your subscription renewal is coming up next month",
            "Technical support ticket has been resolved, please verify",
            "Board meeting minutes are attached for your review",
            "New company policy regarding remote work has been updated",
            "Your flight confirmation for the business trip to Chicago",
            "Please update your emergency contact information in HR system",
            "Congratulations on completing the certification program",
            "The server maintenance is scheduled for this weekend"
        ]
        
        # Spam emails using various concealment/deception tactics
        spam_emails = [
            # Urgency deception
            "URGENT: Your account will be closed in 24 hours! Click here immediately",
            "FINAL NOTICE: Immediate action required to avoid account suspension",
            "ACT NOW: Limited time offer expires in 2 hours, don't miss out",
            
            # Authority impersonation (concealment of true sender)
            "From: Bank Security Team - Verify your account to prevent fraud",
            "IRS Notice: You owe back taxes, pay immediately to avoid legal action",
            "PayPal Security Alert: Suspicious activity detected on your account",
            
            # Obfuscation techniques (concealing spam words)
            "Make m0ney fast with this amaz1ng opportunity, no experience needed",
            "F.R.E.E money, click here for your c@sh prize, you've won big",
            "V1agra and c1alis at lowest prices, discrete shipping worldwide",
            
            # Emotional manipulation
            "You've inherited $2.5 million from a distant relative in Nigeria",
            "Lonely? Find your soulmate tonight, thousands of singles waiting",
            "Your computer is infected! Download our antivirus immediately",
            
            # Fake legitimacy (concealing commercial intent)
            "Congratulations! You've been selected for our exclusive survey reward",
            "Your package delivery failed, click to reschedule delivery",
            "Security update required for your email account, verify now",
            
            # Cryptocurrency/investment scams
            "Bitcoin investment opportunity: Turn $100 into $10000 in 30 days",
            "Secret trading algorithm used by Wall Street, get rich quick",
            "Cryptocurrency giveaway: Send 1 Bitcoin, receive 10 Bitcoin back"
        ]
        
        # Create DataFrame
        emails = legit_emails + spam_emails
        labels = [0] * len(legit_emails) + [1] * len(spam_emails)  # 0 = legit, 1 = spam
        
        return pd.DataFrame({
            'email': emails,
            'label': labels,
            'type': ['Legitimate'] * len(legit_emails) + ['Spam'] * len(spam_emails)
        })
    
    def extract_deception_features(self, text):
        """
        Extract features that indicate concealment/deception tactics
        """
        features = {}
        text_lower = text.lower()
        
        # Urgency indicators
        urgency_words = ['urgent', 'immediate', 'act now', 'limited time', 'expires', 'final notice']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)
        
        # Authority impersonation
        authority_words = ['bank', 'irs', 'paypal', 'security team', 'government', 'police']
        features['authority_score'] = sum(1 for word in authority_words if word in text_lower)
        
        # Obfuscation (character substitution)
        features['obfuscation_score'] = len(re.findall(r'[a-z][0-9@$][a-z]', text_lower))
        
        # Excessive punctuation/caps (attention grabbing)
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['exclamation_count'] = text.count('!')
        
        # Money/prize mentions
        money_words = ['money', 'cash', 'prize', 'win', 'free', 'million', 'thousand', '$']
        features['money_score'] = sum(1 for word in money_words if word in text_lower)
        
        # Suspicious links/actions
        action_words = ['click here', 'download', 'verify', 'confirm', 'update']
        features['action_score'] = sum(1 for word in action_words if word in text_lower)
        
        return features
    
    def train_models(self, df):
        """
        Train bagging ensemble models
        """
        # Vectorize text
        X_text = self.vectorizer.fit_transform(df['email'])
        
        # Extract deception features
        deception_features = []
        for email in df['email']:
            features = self.extract_deception_features(email)
            deception_features.append(list(features.values()))
        
        # Combine text features with deception features
        X_deception = np.array(deception_features)
        X_combined = np.hstack([X_text.toarray(), X_deception])
        
        y = df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train bagging models
        print("Training Bagging with Decision Trees...")
        self.bagging_tree.fit(X_train, y_train)
        
        print("Training Bagging with Naive Bayes...")
        self.bagging_nb.fit(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate and compare bagging models
        """
        # Predictions
        pred_tree = self.bagging_tree.predict(X_test)
        pred_nb = self.bagging_nb.predict(X_test)
        
        print("=== Bagging Decision Tree Results ===")
        print(classification_report(y_test, pred_tree, target_names=['Legitimate', 'Spam']))
        
        print("\n=== Bagging Naive Bayes Results ===")
        print(classification_report(y_test, pred_nb, target_names=['Legitimate', 'Spam']))
        
        # Confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        cm_tree = confusion_matrix(y_test, pred_tree)
        cm_nb = confusion_matrix(y_test, pred_nb)
        
        sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Spam'], 
                   yticklabels=['Legitimate', 'Spam'], ax=axes[0])
        axes[0].set_title('Bagging Decision Tree\nConfusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Reds',
                   xticklabels=['Legitimate', 'Spam'], 
                   yticklabels=['Legitimate', 'Spam'], ax=axes[1])
        axes[1].set_title('Bagging Naive Bayes\nConfusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.suptitle('Spam Detection: Bagging Ensemble Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the confusion matrix figure
        confusion_matrix_filename = 'Lecture_9_CCD_Methods/spam_filter_confusion_matrices.png'
        plt.savefig(confusion_matrix_filename, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices saved as: {confusion_matrix_filename}")
        
        plt.show()
        
        return pred_tree, pred_nb
    
    def analyze_deception_tactics(self, df):
        """
        Analyze the concealment/deception tactics used in spam
        """
        spam_emails = df[df['label'] == 1]['email'].tolist()
        
        all_features = []
        for email in spam_emails:
            features = self.extract_deception_features(email)
            all_features.append(features)
        
        # Aggregate features
        feature_summary = {}
        for feature in all_features[0].keys():
            feature_summary[feature] = [f[feature] for f in all_features]
        
        # Plot deception tactics analysis
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Spam Concealment & Deception Tactics Analysis', fontsize=16, fontweight='bold')
        
        tactics = ['urgency_score', 'authority_score', 'obfuscation_score', 
                  'money_score', 'action_score', 'caps_ratio']
        
        for i, tactic in enumerate(tactics):
            row = i // 3
            col = i % 3
            
            values = feature_summary[tactic]
            axes[row, col].hist(values, bins=5, alpha=0.7, color='red', edgecolor='black')
            axes[row, col].set_title(f'{tactic.replace("_", " ").title()}')
            axes[row, col].set_xlabel('Score')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the deception tactics figure
        deception_tactics_filename = 'Lecture_9_CCD_Methods/spam_deception_tactics_analysis.png'
        plt.savefig(deception_tactics_filename, dpi=300, bbox_inches='tight')
        print(f"Deception tactics analysis saved as: {deception_tactics_filename}")
        
        plt.show()
    
    def demonstrate_concealment_detection(self, df):
        """
        Demonstrate how the filter detects concealed spam tactics
        """
        print("=== CONCEALMENT & DECEPTION DETECTION EXAMPLES ===\n")
        
        spam_examples = df[df['label'] == 1]['email'].tolist()[:5]
        
        for i, email in enumerate(spam_examples, 1):
            print(f"SPAM EXAMPLE {i}:")
            print(f"Email: {email}")
            
            # Extract features
            features = self.extract_deception_features(email)
            print("Detected Deception Tactics:")
            for tactic, score in features.items():
                if score > 0:
                    print(f"  - {tactic.replace('_', ' ').title()}: {score}")
            
            # Get prediction confidence
            email_vectorized = self.vectorizer.transform([email])
            deception_features = np.array([list(features.values())])
            combined_features = np.hstack([email_vectorized.toarray(), deception_features])
            
            prob_tree = self.bagging_tree.predict_proba(combined_features)[0]
            prob_nb = self.bagging_nb.predict_proba(combined_features)[0]
            
            print(f"Bagging Tree Spam Probability: {prob_tree[1]:.3f}")
            print(f"Bagging NB Spam Probability: {prob_nb[1]:.3f}")
            print("-" * 80)

def main():
    """
    Main function to demonstrate spam filter with bagging
    """
    print("="*60)
    print("SPAM DETECTION DEMONSTRATION")
    print("Concealment, Camouflage, and Deception (CCD) Methods")
    print("="*60)
    
    print("\nThis demonstration shows how spam emails use deception tactics")
    print("and how bagging ensemble methods can detect these concealed threats.\n")
    
    # Initialize spam filter
    spam_filter = SpamFilterBagging()
    
    # Create sample data
    print("Creating sample email dataset...")
    df = spam_filter.create_sample_data()
    print(f"Dataset created: {len(df)} emails ({sum(df['label'] == 0)} legitimate, {sum(df['label'] == 1)} spam)")
    
    # Analyze deception tactics
    print("\nAnalyzing spam concealment tactics...")
    spam_filter.analyze_deception_tactics(df)
    
    # Train models
    print("\nTraining bagging ensemble models...")
    X_train, X_test, y_train, y_test = spam_filter.train_models(df)
    
    # Evaluate models
    print("\nEvaluating models...")
    pred_tree, pred_nb = spam_filter.evaluate_models(X_test, y_test)
    
    # Demonstrate concealment detection
    spam_filter.demonstrate_concealment_detection(df)
    
    print("\n=== BAGGING ENSEMBLE ADVANTAGES ===")
    print("1. ROBUSTNESS: Multiple models reduce overfitting to specific spam patterns")
    print("2. DIVERSITY: Different base models catch different deception tactics")
    print("3. STABILITY: Ensemble is less sensitive to training data variations")
    print("4. GENERALIZATION: Better performance on new, unseen spam tactics")
    
    print("\n=== DECEPTION TACTICS DETECTED ===")
    print("1. URGENCY DECEPTION: False time pressure and deadlines")
    print("2. AUTHORITY IMPERSONATION: Pretending to be legitimate organizations")
    print("3. OBFUSCATION: Character substitution to hide spam keywords")
    print("4. EMOTIONAL MANIPULATION: Using fear, greed, and social engineering")
    print("5. FAKE LEGITIMACY: Disguising commercial intent as legitimate communication")
    
    print(f"\n" + "="*60)
    print


if __name__ == "__main__":
    main()