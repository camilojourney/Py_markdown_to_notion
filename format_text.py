from pathlib import Path
import re

def replace_once(text: str, old: str, new: str) -> str:
    return text.replace(old, new, 1)

text = Path('text.md').read_text()

# Top-level heading and part headings
text = replace_once(text, 'Level 2: Machine Learning Algorithms — The Learning Layer', '# Level 2: Machine Learning Algorithms — The Learning Layer\n')
part_headings = [
    ('Part I: The Mathematical Toolkit for Machine Learning', '\n\n## Part I: The Mathematical Toolkit for Machine Learning\n\n'),
    ('Part II: Classical Learning Algorithms', '\n\n## Part II: Classical Learning Algorithms\n\n'),
    ('Part III: The Art and Science of Model Training', '\n\n## Part III: The Art and Science of Model Training\n\n'),
    ('Part IV: Engineering the Input: Feature Craftsmanship', '\n\n## Part IV: Engineering the Input: Feature Craftsmanship\n\n'),
    ('Part V: Architectures for Complex Data - An Introduction to Deep Learning', '\n\n## Part V: Architectures for Complex Data - An Introduction to Deep Learning\n\n'),
    ('Part VI: Tools of the Trade: ML Frameworks', '\n\n## Part VI: Tools of the Trade: ML Frameworks\n\n'),
]
for old, new in part_headings:
    text = replace_once(text, old, new)

# Section headings (H3)
sections = [
    '1.1 Linear Algebra: The Language of Data',
    '1.2 Calculus: The Engine of Optimization',
    '1.3 Probability & Statistics: The Framework for Uncertainty',
    '2.1 Supervised Learning: Learning from Labels',
    '2.1.1 Linear Regression',
    '2.1.2 Logistic Regression',
    '2.1.3 K-Nearest Neighbors (KNN)',
    '2.1.4 Support Vector Machines (SVM)',
    '2.1.5 Decision Trees',
    '2.1.6 Random Forests',
    '2.2 Unsupervised Learning: Finding Structure in Data',
    '2.2.1 K-Means Clustering',
    '2.2.2 Principal Component Analysis (PCA)',
    '3.1 Optimization: Finding the Best Model',
    '3.2 Evaluation: Measuring Model Performance',
    '3.3 Validation: Ensuring Robustness',
    '4.1 Handling Imperfect Data',
    '4.2 Transforming Features',
    '5.1 Convolutional Neural Networks (CNNs): The Vision Specialists',
    '5.2 Recurrent Neural Networks (RNNs): The Sequence Masters',
    '5.3 Transformers: The New Paradigm in NLP',
    '6.1 Scikit-Learn: The Go-To for Classical ML',
    '6.2 The Deep Learning Ecosystem: TensorFlow and PyTorch',
]
for sec in sections:
    text = replace_once(text, sec, f'\n\n### {sec}\n\n')

# Subsection emphasis
subsections = [
    'Vectors, Matrices, and Tensors: The Building Blocks of Data',
    'Core Operations and Their Role in Learning',
    'Linear Transformations: Manipulating Data Space',
    'Eigenvectors and Eigenvalues: Uncovering the Structure of Data',
    'Derivatives, Gradients, and the Loss Landscape',
    'The Chain Rule and Backpropagation',
    'Key Distributions in Machine Learning',
    'Bayesian Inference and Estimation Theory',
    'The Role of Loss Functions',
    'Gradient Descent Deep Dive',
    'Advanced Optimizers: Adam',
    'The Peril of Overfitting',
    'Cross-Validation',
    'Feature Scaling',
    'Interaction and Polynomial Features',
    'Core Architectural Components',
    'Challenges',
]
for label in subsections:
    text = text.replace(label, f'\n\n**{label}**\n\n')

# Ensure key colon phrases start on new line and are bold (no bullet conversion)
colon_terms = [
    'Scalars:', 'Vectors:', 'Matrices:', 'Dot Product:', 'Matrix Multiplication:',
    'Scaling:', 'Rotation:', 'Translation:', 'Eigenvectors ($v$):', 'Eigenvalues ($\\lambda$):',
    'Intuition:', 'Mathematical Formulation:', 'Hypothesis:', 'Cost Function:', 'Optimization:',
    'Gradient Descent:', 'Normal Equation:', 'Python Implementation (Scikit-Learn):',
    'Real-World Applications:', 'Derivatives:', 'Gradients:', 'Momentum:',
    'Adaptive Learning Rates (from RMSprop):', 'Pros:', 'Cons:', 'Initialization:',
    'Assignment Step:', 'Update Step:', 'Gaussian (Normal) Distribution:',
    'Bernoulli Distribution:', 'Binomial Distribution:', 'Maximum Likelihood Estimation (MLE):',
    'Maximum A Posteriori (MAP) Estimation:', 'Mean Absolute Error (MAE):',
    'Mean Squared Error (MSE):', 'Root Mean Squared Error (RMSE):', 'R-squared ($R^2$):',
    'Complete Case Analysis (CCA):', 'Mean/Median/Mode Imputation:',
    'Label Encoding:', 'One-Hot Encoding:', 'Normalization (Min-Max Scaling):',
    'Standardization (Z-score Normalization):', 'Convolutional Layer:', 'Pooling Layer:',
    'Fully Connected Layer:', 'Recurrent Connection:', 'Shared Weights:',
    'Self-Attention Mechanism:', 'Multi-Head Attention:', 'Consistent API:',
    'Comprehensive Toolset:', 'Excellent Documentation:', 'Tensor-based Computation:',
    'Automatic Differentiation:', 'GPU and TPU Integration:',
]
for term in colon_terms:
    text = text.replace(term, f'\n**{term}** ')

# Fix specific concatenated cases (Intuition sections)
concat_cases = [
    ('Linear Regression** Intuition:', 'Linear Regression\n\n**Intuition:** '),
    ('Logistic Regression** Intuition:', 'Logistic Regression\n\n**Intuition:** '),
    ('K-Nearest Neighbors (KNN)** Intuition:', 'K-Nearest Neighbors (KNN)\n\n**Intuition:** '),
    ('Support Vector Machines (SVM)** Intuition:', 'Support Vector Machines (SVM)\n\n**Intuition:** '),
    ('Decision Trees** Intuition:', 'Decision Trees\n\n**Intuition:** '),
    ('Random Forests** Intuition:', 'Random Forests\n\n**Intuition:** '),
    ('K-Means Clustering** Intuition:', 'K-Means Clustering\n\n**Intuition:** '),
    ('Principal Component Analysis (PCA)** Intuition:', 'Principal Component Analysis (PCA)\n\n**Intuition:** '),
    ('Convolutional Neural Networks (CNNs): The Vision Specialists\n\n**Intuition:**', 'Convolutional Neural Networks (CNNs): The Vision Specialists\n\n**Intuition:**'),
    ('Recurrent Neural Networks (RNNs): The Sequence Masters\n\n**Intuition:**', 'Recurrent Neural Networks (RNNs): The Sequence Masters\n\n**Intuition:**'),
    ('Transformers: The New Paradigm in NLP\n\n**Intuition:**', 'Transformers: The New Paradigm in NLP\n\n**Intuition:**'),
]
for old, new in concat_cases:
    text = text.replace(old, new)

# Code blocks
code_snippets = {
    """Pythonfrom sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assume X (features) and y (target) are pre-loaded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Intercept (theta_0): {model.intercept_}")
print(f"Coefficients (theta_1,...): {model.coef_}")""": """Python
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assume X (features) and y (target) are pre-loaded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Intercept (theta_0): {model.intercept_}")
print(f"Coefficients (theta_1,...): {model.coef_}")
```
""",
    """Pythonfrom sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics = accuracy_score, classification_report

# Assume X (features) and y (binary target) are pre-loaded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))""": """Python
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Assume X (features) and y (binary target) are pre-loaded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
```
""",
}

# fix typo 'metrics ='
text = text.replace('from sklearn.metrics = accuracy_score', 'from sklearn.metrics import accuracy_score')

for old, new in code_snippets.items():
    text = text.replace(old, new)

# Additional code blocks (KNN, SVM, etc.) can be handled similarly
# For brevity, use regex to wrap sequences starting with 'Pythonfrom' up to blank line before 'Real-World Applications'

def wrap_code(match: re.Match) -> str:
    content = match.group(1)
    return f"Python\n```python\n{content.strip()}\n```\n"

text = re.sub(
    r"Pythonfrom([\s\S]*?)(?=\n\*\*Real-World Applications:\*\*|\n\n[A-Z0-9]|\Z)",
    lambda m: wrap_code(m),
    text
)

# Convert problematic $$ usage to proper form
text = text.replace('range $$, which', 'range $[0, 1]$ (`$$`), which')
text = text.replace('range, typically $$', 'range, typically $[0, 1]$ (`$$`)')


def convert_dollars(match: re.Match) -> str:
    expr = match.group(1).strip()
    return f"\n$$\n{expr}\n$$\n"

text = re.sub(r"\$\$(.+?)\$\$", convert_dollars, text, flags=re.S)

# Gradient descent characteristics table
text = text.replace(
    'CharacteristicBatch Gradient DescentStochastic Gradient Descent (SGD)Mini-Batch Gradient DescentBatch SizeFull Dataset (N)1n (where 1 < n < N)Update FrequencyOnce per epochFor every training exampleFor every mini-batchComputational CostVery HighLowMediumConvergence StabilitySmooth, direct convergenceNoisy, may oscillate around minimumRelatively smooth with some noiseMemory RequirementHighLowMediumBest ForSmall datasets, convex loss functionsLarge datasets, non-convex problemsThe default choice for most modern applications',
    '| Characteristic | Batch Gradient Descent | Stochastic Gradient Descent (SGD) | Mini-Batch Gradient Descent |\n|----------------|-------------------------|-------------------------------------|------------------------------|\n| Batch Size | Full Dataset (N) | 1 | n (where 1 < n < N) |\n| Update Frequency | Once per epoch | For every training example | For every mini-batch |\n| Computational Cost | Very High | Low | Medium |\n| Convergence Stability | Smooth, direct convergence | Noisy, may oscillate around minimum | Relatively smooth with some noise |\n| Memory Requirement | High | Low | Medium |\n| Best For | Small datasets, convex loss functions | Large datasets, non-convex problems | The default choice for most modern applications |
)

# Classification metrics formatting
text = text.replace('Accuracy =', '**Accuracy =**')
text = text.replace('Precision =', '\n**Precision =**')
text = text.replace('Recall =', '\n**Recall =**')
text = text.replace('F1 =', '\n**F1 =**')

# Regression metrics formatting
reg_terms = ['Mean Absolute Error (MAE):', 'Mean Squared Error (MSE):', 'Root Mean Squared Error (RMSE):', 'R-squared ($R^2$):']
for term in reg_terms:
    text = text.replace(term, f'\n**{term}** ')

# Algorithm comparison table
text = text.replace(
    'AlgorithmTypeCore IdeaInterpretabilityKey StrengthsKey WeaknessesRequires Scaling?Linear RegressionSupervised (Regression)Fit a line/hyperplane to dataHighSimple, fast, interpretable coefficientsAssumes linearity, sensitive to outliersYes (for regularization)Logistic RegressionSupervised (Classification)Fit a sigmoid curve to dataHighProbabilistic output, fast, interpretableAssumes linearity of log-oddsYes (for regularization)K-Nearest NeighborsSupervised (Class./Reg.)Majority vote of K closest neighborsMediumSimple, non-parametric, no training phaseSlow at inference, curse of dimensionalityYesSupport Vector MachineSupervised (Class./Reg.)Find the maximum-margin hyperplaneLow (with kernels)Effective in high dimensions, kernel trickComputationally expensive, sensitive to paramsYesDecision TreeSupervised (Class./Reg.)Learn a hierarchy of if-then rulesVery HighEasy to understand and visualize, handles non-linearityProne to overfitting, unstableNoRandom ForestSupervised (Class./Reg.)Ensemble of decorrelated decision treesLowHigh accuracy, robust to overfitting, handles missing dataLess interpretable than a single treeNoK-Means ClusteringUnsupervised (Clustering)Partition data into K groups by minimizing inertiaHighSimple, efficient, scalableMust specify K, sensitive to initial centroidsYesPrincipal Component AnalysisUnsupervised (Dim. Red.)Find orthogonal axes of maximum varianceMediumReduces noise, decorrelates features, aids visualizationAssumes linear projections, can lose informationYes',
    '| Algorithm | Type | Core Idea | Interpretability | Key Strengths | Key Weaknesses | Requires Scaling? |\n|-----------|------|-----------|------------------|---------------|----------------|-------------------|\n| Linear Regression | Supervised (Regression) | Fit a line/hyperplane to data | High | Simple, fast, interpretable coefficients | Assumes linearity, sensitive to outliers | Yes (for regularization) |\n| Logistic Regression | Supervised (Classification) | Fit a sigmoid curve to data | High | Probabilistic output, fast, interpretable | Assumes linearity of log-odds | Yes (for regularization) |\n| K-Nearest Neighbors | Supervised (Class./Reg.) | Majority vote of K closest neighbors | Medium | Simple, non-parametric, no training phase | Slow at inference, curse of dimensionality | Yes |\n| Support Vector Machine | Supervised (Class./Reg.) | Find the maximum-margin hyperplane | Low (with kernels) | Effective in high dimensions, kernel trick | Computationally expensive, sensitive to params | Yes |\n| Decision Tree | Supervised (Class./Reg.) | Learn a hierarchy of if-then rules | Very High | Easy to understand and visualize, handles non-linearity | Prone to overfitting, unstable | No |\n| Random Forest | Supervised (Class./Reg.) | Ensemble of decorrelated decision trees | Low | High accuracy, robust to overfitting, handles missing data | Less interpretable than a single tree | No |\n| K-Means Clustering | Unsupervised (Clustering) | Partition data into K groups by minimizing inertia | High | Simple, efficient, scalable | Must specify K, sensitive to initial centroids | Yes |\n| Principal Component Analysis | Unsupervised (Dim. Red.) | Find orthogonal axes of maximum variance | Medium | Reduces noise, decorrelates features, aids visualization | Assumes linear projections, can lose information | Yes |
)

# Conclusion heading
text = replace_once(text, '\n\n## Conclusion', '\n\n## Conclusion\n\n')
text = text.replace('ConclusionThis', '\n\n## Conclusion\n\nThis')

Path('text_formatted.md').write_text(text)
