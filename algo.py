from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Load the newly uploaded dataset
crop_df = pd.read_csv("Crop_recommendation.csv")

# Display basic info and the first few rows
crop_df.info(), crop_df.head()
# Encode the target labels
le = LabelEncoder()
crop_df['label_encoded'] = le.fit_transform(crop_df['label'].str.strip())

# Features and target
X = crop_df.drop(['label', 'label_encoded'], axis=1)
y = crop_df['label_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LogisticRegression(max_iter=1000)
dt_model = DecisionTreeClassifier(random_state=42)

# Train models
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Predict
lr_preds = lr_model.predict(X_test)
dt_preds = dt_model.predict(X_test)

# Collect metrics
def evaluate_model(name, y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1 Score": report["weighted avg"]["f1-score"]
    }

# Evaluate both models
lr_metrics = evaluate_model("Logistic Regression", y_test, lr_preds)
dt_metrics = evaluate_model("Decision Tree", y_test, dt_preds)

# Show confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix - Logistic Regression")

sns.heatmap(confusion_matrix(y_test, dt_preds), annot=True, fmt='d', ax=axes[1], cmap="Greens")
axes[1].set_title("Confusion Matrix - Decision Tree")

plt.tight_layout()
plt.show()

# Create a comparison table
comparison_df = pd.DataFrame([lr_metrics, dt_metrics])
comparison_df
