import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the new confusion matrix
new_confusion_matrix = np.array([
    [148, 39, 0, 0, 0],
    [45, 163, 0, 0, 0],
    [20, 52, 109, 11, 0],
    [0, 0, 7, 166, 0],
    [0, 0, 0, 0, 182]
])

# Define the new labels
new_labels = ['0.5g', '1g', '1.5g', '2g', '2.5g']

# Create a heatmap of the new confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(new_confusion_matrix, annot=True, fmt='d', cmap='YlOrBr', cbar=False,
            xticklabels=new_labels, yticklabels=new_labels, annot_kws={"weight": "bold", "size": 12},
            cbar_kws={'label': 'Scale'})

plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.show()
