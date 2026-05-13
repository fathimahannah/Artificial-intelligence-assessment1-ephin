# Import necessary libraries 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, precision_score, f1_score, accuracy_score
import seaborn as sns

# Load and clean data
df = pd.read_csv('indian_food.csv')
df.columns = df.columns.str.strip()
df = df[df['flavor_profile'] != '-1']
print(df.head(10))

# Rule 1: Sweet Dish
def rule1(row):
    if row['flavor_profile'] == 'sweet' and row['course'] == 'dessert':
        return 'Sweet Dish'
    return 'Neutral'

# Rule 2: Savory Dish
def rule2(row):
    if row['flavor_profile'] == 'spicy' and row['course'] == 'main course':
        return 'Savory Dish'
    return 'Neutral'

# Apply rules
df['rule1'] = df.apply(rule1, axis=1)
df['rule2'] = df.apply(rule2, axis=1)

# Final classification
def final_class(row):
    if row['rule1'] == 'Sweet Dish':
        return 'Sweet Dish'
    elif row['rule2'] == 'Savory Dish':
        return 'Savory Dish'
    return 'Neutral'

df['classification'] = df.apply(final_class, axis=1)

# --- Binary labels for evaluation (Sweet Dish detection) ---
df['true_label'] = (df['flavor_profile'] == 'sweet').astype(int)
df['predicted_score'] = (df['rule1'] == 'Sweet Dish').astype(int)

# --- Confusion Matrix ---
cm = confusion_matrix(df['true_label'], df['predicted_score'])

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('Confusion Matrix - Sweet Dish Classification', color='darkred')
plt.xlabel('Predicted Label', color='darkred')
plt.ylabel('True Label', color='darkred')
plt.tight_layout()
plt.show()

# --- ROC and AUC ---
fpr, tpr, _ = roc_curve(df['true_label'], df['predicted_score'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
plt.xlabel('False Positive Rate', color='darkred')
plt.ylabel('True Positive Rate', color='darkred')
plt.title('ROC Curve for Sweet Dish Classification', color='darkred')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Performance Metrics ---
accuracy = accuracy_score(df['true_label'], df['predicted_score'])
precision = precision_score(df['true_label'], df['predicted_score'])
recall = recall_score(df['true_label'], df['predicted_score'])
f1 = f1_score(df['true_label'], df['predicted_score'])


print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-Score : {f1:.3f}")
print(f"AUC Score: {roc_auc:.3f}")

print("\nConfusion Matrix:")
print(cm)


