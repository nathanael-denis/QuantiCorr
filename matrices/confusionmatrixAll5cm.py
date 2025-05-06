import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the original confusion matrix
original_confusion_matrix = np.array([
    [84,  1,  0,  4,  0,  0,  0,  2,  0],
    [ 2, 38,  0,  7,  0, 14,  4, 22,  0],
    [ 0,  0, 81,  0,  0,  0,  0,  0,  0],
    [ 1, 10,  0, 48,  0,  0,  0,  6,  0],
    [ 0,  0,  0,  0, 88,  0,  0,  0,  0],
    [ 0, 17,  0,  2,  0, 29,  3, 15,  0],
    [ 0,  7,  0,  0,  0,  3, 65,  0,  0],
    [ 1, 18,  0, 15,  0, 15,  0, 25,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0, 89]
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
plt.figure(figsize=(8, 16))  # Adjusted figure size to be even more vertical
sns.heatmap(rearranged_confusion_matrix, annot=True, fmt='d', cmap='YlOrBr', cbar=False,
            xticklabels=desired_labels, yticklabels=desired_labels, annot_kws={"size": 18, "weight": "bold"})

plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=20)
plt.title('Confusion Matrix', fontsize=24, pad=30)  # Increased padding to the title
plt.xticks(rotation=45, ha='right', fontsize=18)
plt.yticks(rotation=0, fontsize=18)

plt.tight_layout(pad=5)  # Increased padding to the layout
plt.show()
