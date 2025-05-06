import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the new confusion matrix
new_confusion_matrix = np.array([
    [332, 31, 11, 0, 0],
    [37, 362, 17, 0, 0],
    [6, 14, 364, 0, 0],
    [0, 0, 11, 335, 0],
    [0, 0, 0, 0, 364]
])

# Define the labels for the new confusion matrix
new_labels = ['0.5g', '1g', '1.5g', '2g', '2.5g']

# Create a heatmap of the new confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(new_confusion_matrix, annot=True, fmt='d', cmap='YlOrBr', cbar=False,
            xticklabels=new_labels, yticklabels=new_labels, annot_kws={"weight": "bold", "size": 14})

# Increase the size of the axis labels and title
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)
plt.title('Confusion Matrix', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(rotation=0, fontsize=14)

plt.show()
