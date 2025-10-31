import matplotlib.pyplot as plt
import numpy as np

# Data for Direct Line-of-Sight Quantification (Scenario 1)
rounds = np.arange(1, 11)
f1_scores_s1 = np.array([0.8144, 0.8363, 0.8097, 0.8118, 0.8090, 0.8090, 0.8127, 0.7688, 0.8394, 0.8394])
mean_f1_s1 = np.mean(f1_scores_s1)
std_f1_s1 = np.std(f1_scores_s1)

# Plot
plt.figure(figsize=(8,5))
plt.plot(rounds, f1_scores_s1, marker='o', color='darkorange', linewidth=2, label='F1-score per round')
plt.axhline(mean_f1_s1, color='gray', linestyle='--', linewidth=1.5, label=f'Average = {mean_f1_s1:.3f}')
plt.fill_between(rounds, mean_f1_s1 - std_f1_s1, mean_f1_s1 + std_f1_s1, color='goldenrod', alpha=0.2, label='±1 standard deviation')

# Labels and formatting
plt.xlabel('Round', fontsize=14)
plt.ylabel('F1-score', fontsize=14)
#plt.title('F1-score per Round – Direct Line-of-Sight Quantification', fontsize=15, weight='bold')
plt.xticks(rounds, fontsize=12)
plt.yticks(np.linspace(0.76, 0.84, 9), fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.show()
