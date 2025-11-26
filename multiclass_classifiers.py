"""
Best Multi-Class Classification Models
Comparison of top performing classifiers for multi-class tasks
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification
import numpy as np

# Generate sample multi-class data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, 
                          n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best Multi-Class Classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

print("Multi-Class Classification Performance Comparison:")
print("="*60)

best_model = None
best_score = 0

for name, clf in classifiers.items():
    # Train model
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = name

print(f"\nBest Model: {best_model} with accuracy: {best_score:.4f}")

# Recommended replacement for LogisticRegression
print("\n" + "="*60)
print("RECOMMENDATION FOR MULTI-CLASS TASKS:")
print("="*60)
print("1. RandomForest - Best overall choice")
print("   - Handles multiple classes naturally")
print("   - Robust to overfitting")
print("   - Feature importance")
print("   - Works well with mixed data types")

print("\n2. GradientBoosting - High performance")
print("   - Often highest accuracy")
print("   - Good for complex patterns")
print("   - Requires more tuning")

print("\n3. SVM - Good for high-dimensional data")
print("   - Effective with many features")
print("   - Memory efficient")
print("   - Slower on large datasets")

# Example replacement code
print("\n" + "="*60)
print("REPLACE YOUR LOGISTIC REGRESSION WITH:")
print("="*60)
print("""
# OLD CODE:
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()

# NEW CODE (RECOMMENDED):
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Alternative high-performance option:
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
""")