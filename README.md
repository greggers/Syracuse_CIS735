# Syracuse University CIS 735: Machine Learning for Cyber Security

Welcome to the official repository for Syracuse University's CIS 735 course: Machine Learning for Cyber Security. This repository contains source code, examples, and resources to help students understand the intersection of machine learning and cybersecurity.

## Course Overview

This course explores how machine learning techniques can be applied to cybersecurity challenges, as well as how to secure machine learning systems themselves. Through a combination of theoretical foundations and practical applications, students will develop the skills needed to implement and evaluate machine learning solutions for security problems.

## Learning Objectives

After taking this course, students will be able to:

- Understand basic principles of machine learning
- Understand threats to cybersecurity systems
- Understand how to detect and circumvent threats and attacks to systems using machine learning algorithms
- Understand how machine learning algorithms may be used to build cybersecurity systems
- Understand how to build secure machine learning algorithms (develop a basic understanding of how security of machine learning algorithms may be improved by including security constructs in the algorithms)
- Develop an understanding of interaction of machine learning and cyber security systems
- Develop an understanding of and build from current research literature in cybersecurity and machine learning

## Course Materials

### [Lecture 2: Bayes Theorem and Distributions](./Lecture_2_Bayes_Theorem_and_Distributions)
- Bayes Theorem examples (binary and continuous)
- Parametric density functions
- Non-parametric density estimation
- Semi-parametric density estimation with mixture models

### [Lecture 3: Distance Measures](./Lecture_3_Distance_Measures)
- Euclidean, Manhattan, and Minkowski distances
- Mahalanobis distance for correlated features
- Cosine similarity for text and high-dimensional data
- Jaccard similarity for sets
- Hamming distance for binary vectors
- Edit distance for strings
- Applications of distance metrics in machine learning algorithms

### [Lecture 4: Features and Dimensionality](./Lecture_4_Features_and_Dimensionality)
- The curse of dimensionality and its implications for machine learning
- Feature extraction and selection techniques
- Dimensionality reduction methods:
  - Principal Component Analysis (PCA) for unsupervised dimensionality reduction
  - Linear Discriminant Analysis (LDA) for supervised dimensionality reduction
  - Autoencoders for non-linear dimensionality reduction and feature learning
- Visualization of high-dimensional data
- Impact of dimensionality on model performance and computational efficiency

### [Lecture 5: Support Vector Machines](./Lecture_5_Support_Vector_Machines)
- Linear and non-linear SVM classification
- Kernel methods and the kernel trick
- Support vector selection and margin optimization
- SVM for cybersecurity applications:
  - Malware detection and classification
  - Network intrusion detection
  - Anomaly detection in system behavior
- Handling imbalanced datasets in security contexts
- SVM parameter tuning and regularization

### [Lecture 6: Neural Networks](./Lecture_6_Neural_Networks)
- Fundamentals of artificial neural networks
- Feedforward networks and backpropagation
- Deep learning architectures:
  - Convolutional Neural Networks (CNNs) for image-based security analysis
  - Recurrent Neural Networks (RNNs) for sequential data analysis
  - Long Short-Term Memory (LSTM) networks for time-series security data
- Neural networks in cybersecurity:
  - Automated threat detection
  - Behavioral analysis and user authentication
  - Network traffic analysis
  - Advanced persistent threat (APT) detection
- Training strategies and optimization techniques

### [Lecture 7: Missing Data](./Lecture_7_Missing_Data)
- Types of missing data: MCAR, MAR, and MNAR
- Impact of missing data on machine learning models
- Missing data handling techniques:
  - Deletion methods (listwise, pairwise)
  - Imputation methods (mean, median, mode, regression-based)
  - Advanced imputation (multiple imputation, iterative imputation)
  - Machine learning-based imputation
- Missing data in cybersecurity contexts:
  - Incomplete log files and network data
  - Sensor failures in IoT security systems
  - Partial attack signatures and incomplete threat intelligence
- Evaluation of imputation quality and model robustness

### [Lecture 8: Model Selection](./Lecture_8_Model_Selection)
- Cross-validation techniques and their applications
- Hyperparameter optimization strategies:
  - Grid search and random search
  - Bayesian optimization
  - Evolutionary algorithms for parameter tuning
- Model evaluation metrics for cybersecurity:
  - Precision, recall, and F1-score in threat detection
  - ROC curves and AUC for binary classification
  - Cost-sensitive evaluation for security applications
- Ensemble methods and model combination:
  - Bagging and boosting techniques
  - Stacking and voting classifiers
- Bias-variance tradeoff in security model selection
- Model interpretability and explainability in cybersecurity

### [Lecture 9: Camouflage, Concealment and Deception (CCD) Methods](./Lecture_9_CCD_Methods)
- Introduction to deception in cybersecurity
- Camouflage techniques:
  - Steganography and data hiding methods
  - Traffic obfuscation and protocol camouflage
  - Code obfuscation and anti-analysis techniques
- Concealment strategies:
  - Covert channels and hidden communication
  - Data exfiltration concealment methods
  - Rootkit and malware hiding techniques
- Deception technologies:
  - Honeypots and honeynets for threat intelligence
  - Decoy systems and fake services
  - Deceptive defense mechanisms
- Machine learning applications in CCD:
  - Automated deception detection
  - Adversarial examples and model deception
  - Spam filtering and social engineering detection
- Ethical considerations and defensive applications

### [Lecture 10: Advanced ML Applications in Cybersecurity](./Lecture_10_Advanced_Applications)
- Real-world machine learning implementations for cybersecurity
- Advanced threat detection systems:
  - Multi-stage attack detection using ensemble methods
  - Behavioral analytics for insider threat detection
  - Zero-day exploit detection using anomaly detection
  - Advanced Persistent Threat (APT) identification
- Graph Neural Networks (GNNs) for cybersecurity:
  - Network topology analysis and anomaly detection
  - Graph-based malware detection and classification
  - Social network analysis for threat intelligence
  - Enterprise network security monitoring with GNNs
  - Node-level and graph-level anomaly detection
  - Graph Attention Networks (GAT) for cybersecurity applications
- Time-series analysis for security:
  - Network traffic anomaly detection
  - Log analysis and event correlation
  - Predictive security analytics
- Natural Language Processing (NLP) in cybersecurity:
  - Threat intelligence extraction from unstructured data
  - Phishing email detection and classification
  - Malware analysis through code similarity
  - Social engineering attack detection
- Computer vision applications:
  - CAPTCHA solving and security implications
  - Visual malware analysis and classification
  - Biometric security systems
- Federated learning for distributed security:
  - Privacy-preserving threat detection
  - Collaborative security without data sharing
  - Distributed intrusion detection systems
- Adversarial machine learning:
  - Attacks against ML-based security systems
  - Adversarial examples in cybersecurity contexts
  - Robust ML model development
  - Defense mechanisms against adversarial attacks
- Case studies and practical implementations:
  - Enterprise security monitoring systems
  - Cloud security analytics platforms
  - IoT device security using ML
  - Mobile security and malware detection
- Digital thread tracking and security analytics:
  - End-to-end security monitoring
  - Attack path reconstruction
  - Incident response automation
  - Security metrics and KPI development

## Course Schedule

| Week | Topic |
|------|-------|
| Week 1 | Introduction to Machine Learning in Security |
| Week 2 | Pattern Recognition and Classification |
| Week 3 | Distance measures |
| Week 4 | Features definition, extraction, and reduction |
| Week 5 | Support Vector Machines |
| Week 6 | Neural Networks |
| Week 7 | Missing Data Handling |
| Week 8 | Model Selection and Evaluation |
| Week 9 | Camouflage, Concealment and Deception Methods |
| Week 10 | Advanced ML Applications and Case Studies |

## Getting Started

To use the code in this repository:

1. Clone the repository:
```bash
git clone https://github.com/greggers/Syracuse_CIS735.git
```

2. Install the required dependencies:
```bash
pip install numpy scipy matplotlib scikit-learn torch torchvision pandas seaborn pillow torch-geometric networkx
```

3. Navigate to specific lecture directories to run examples.

## Repository Structure

```
Syracuse_CIS735/
├── README.md
├── Lecture_2_Bayes_Theorem_and_Distributions/
├── Lecture_3_Distance_Measures/
├── Lecture_4_Features_and_Dimensionality/
├── Lecture_5_Support_Vector_Machines/
├── Lecture_6_Neural_Networks/
├── Lecture_7_Missing_Data/
├── Lecture_8_Model_Selection/
├── Lecture_9_CCD_Methods/
│   ├── steganography.py
│   ├── honey_pot.py
│   ├── spam_filter_bagging.py
│   └── sample_images/
├── Lecture_10_Advanced_Applications/
│   ├── advanced_threat_detection.py
│   ├── behavioral_analytics.py
│   ├── time_series_security.py
│   ├── nlp_security_applications.py
│   ├── federated_security_learning.py
│   ├── adversarial_ml_security.py
│   ├── gnn_node_detection_demo.py
│   ├── enterprise_security_monitoring.py
│   ├── attack_path_reconstruction.py
│   └── security_analytics_dashboard.py
└── requirements.txt
```

## Key Features

- **Practical Examples**: Real-world cybersecurity applications of machine learning
- **Interactive Demonstrations**: Hands-on code examples with visualizations
- **Security Focus**: All examples tailored to cybersecurity contexts
- **Modern Techniques**: Coverage of both classical and deep learning approaches
- **Graph Neural Networks**: Advanced GNN implementations for network security analysis
- **Digital Thread Tracking**: Comprehensive security event monitoring and analysis
- **Adversarial ML**: Understanding and defending against attacks on ML systems
- **Ethical Considerations**: Discussion of responsible AI in security applications

## Advanced Applications Highlights

### Graph Neural Networks for Cybersecurity
The course includes comprehensive coverage of Graph Neural Networks (GNNs) applied to cybersecurity challenges:

- **Network Topology Analysis**: Using GNNs to understand and monitor enterprise network structures
- **Node-Level Anomaly Detection**: Identifying compromised devices and suspicious network entities
- **Graph-Level Classification**: Detecting malicious network patterns and attack signatures
- **Graph Attention Networks**: Advanced attention mechanisms for focusing on critical network components
- **Real-time Security Monitoring**: Scalable GNN implementations for live network analysis
- **Digital Thread Integration**: Tracking security events across graph-structured data

### Practical Security Applications
Students will work with realistic cybersecurity scenarios including:

- **Enterprise Network Security**: Multi-layered defense systems using ensemble ML methods
- **Insider Threat Detection**: Behavioral analytics and anomaly detection for internal threats
- **APT Detection**: Advanced persistent threat identification using time-series analysis
- **Threat Intelligence**: NLP-based extraction and analysis of security intelligence
- **IoT Security**: Specialized ML approaches for Internet of Things device protection
- **Cloud Security Analytics**: Scalable ML solutions for cloud-based security monitoring

## Contributing

Students and the public are encouraged to contribute to this repository by submitting pull requests with improvements, bug fixes, or additional examples.

## Prerequisites

- Basic understanding of Python programming
- Fundamental knowledge of statistics and linear algebra
- Introduction to cybersecurity concepts
- Familiarity with machine learning basics (recommended)
- Understanding of network concepts and graph theory (for advanced topics)

## Contact

For questions or further information about the course, please contact the course instructor.

---

*This repository is maintained for educational purposes as part of Syracuse University's CIS 735 course.*
