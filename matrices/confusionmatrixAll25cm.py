import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the original confusion matrix
original_confusion_matrix = np.array([
    [148, 0, 13, 9, 1, 0, 0, 0, 0],
    [0, 116, 1, 2, 3, 4, 0, 1, 63],
    [98, 0, 117, 3, 0, 0, 0, 1, 0],
    [8, 3, 0, 190, 40, 0, 0, 0, 11],
    [5, 14, 0, 25, 127, 0, 0, 0, 5],
    [0, 3, 1, 0, 0, 123, 44, 1, 22],
    [0, 0, 0, 0, 0, 16, 151, 21, 0],
    [0, 0, 0, 0, 0, 4, 41, 111, 0],
    [1, 13, 0, 2, 5, 60, 16, 2, 119]
])

# Define the original labels
original_labels = ['0.5', '1.5spread', '1.5stack', '1spread', '1stack', '2.5spread', '2.5stack', '2spread', '2stack']

# Define the desired order of labels
desired_labels = ['0.5', '1spread', '1stack', '1.5spread', '1.5stack', '2spread', '2stack', '2.5spread', '2.5stack']

# Create a mapping from original labels to their indices
label_to_index = {label: idx for idx, label in enumerate(original_labels)}

# Create a new confusion matrix with the desired order
rearranged_confusion_matrix = np.zeros_like(original_confusion_matrix)
for i, new_label in enumerate(desired_labels):
    for j, new_label_col in enumerate(desired_labels):
        rearranged_confusion_matrix[i, j] = original_confusion_matrix[label_to_index[new_label], label_to_index[new_label_col]]

# Create a heatmap of the rearranged confusion matrix with increased font sizes
plt.figure(figsize=(8, 16))  # Adjusted figure size to be more vertical
sns.heatmap(rearranged_confusion_matrix, annot=True, fmt='d', cmap='YlOrBr', cbar=False,
            xticklabels=desired_labels, yticklabels=desired_labels, annot_kws={"size": 18, "weight": "bold"})

plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=20)
plt.title('Confusion Matrix', fontsize=24, pad=30)  # Increased padding to the title
plt.xticks(rotation=45, ha='right', fontsize=18)
plt.yticks(rotation=0, fontsize=18)

plt.tight_layout(pad=5)  # Increased padding to the layout
plt.show()
