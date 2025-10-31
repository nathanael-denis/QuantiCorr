import matplotlib.pyplot as plt
import numpy as np

# Data from your table
rounds = np.arange(1, 11)
f1_scores = np.array([0.9254, 0.9327, 0.9327, 0.9184, 0.9343, 0.9343, 0.9315, 0.9375, 0.9375, 0.9294])

# Compute average and variance
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

# Plot
plt.figure(figsize=(8,5))
plt.plot(rounds, f1_scores, marker='o', color='darkorange', linewidth=2, label='F1-score per round')
plt.axhline(mean_f1, color='gray', linestyle='--', linewidth=1.5, label=f'Average = {mean_f1:.3f}')
plt.fill_between(rounds, mean_f1 - std_f1, mean_f1 + std_f1, color='goldenrod', alpha=0.2, label='±1 standard deviation')

# Labels and formatting
#plt.title('F1-score per Training Round – Internal Oxidation Scenario', fontsize=15, weight='bold')
plt.xlabel('Round', fontsize=14)
plt.ylabel('F1-score', fontsize=14)
plt.xticks(rounds, fontsize=12)
plt.yticks(np.linspace(0.91, 0.94, 7), fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
