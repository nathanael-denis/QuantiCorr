import matplotlib.pyplot as plt
import numpy as np

# Models and datasets
models = ['Classifier', 'Regressor']
datasets = ['Test', 'Interpolation']

# Average metrics from your 10-run data
# Classifier
f1_test = 0.99594
f1_interp = 0.39313

# Regressor
rmse_test = 0.027
r2_test = 0.99539
rmse_interp = 0.16149
r2_interp = 0.6597

# Positions for grouped bars
x = np.arange(len(datasets))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8,5))

# Classifier bars (F1)
ax1.bar(x - width/2, [f1_test, f1_interp], width, label='Classifier F1', color='tab:blue')
# Regressor bars (R²)
ax1.bar(x + width/2, [r2_test, r2_interp], width, label='Regressor R²', color='tab:green')

ax1.set_ylabel('F1 / R²')
ax1.set_ylim(0,1.1)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.set_title('Comparison of Classification and Regression Performance')
ax1.legend(loc='upper left')

# Secondary axis for RMSE
ax2 = ax1.twinx()
ax2.plot(x, [rmse_test, rmse_interp], 'o--', color='tab:red', label='Regressor RMSE')
ax2.set_ylabel('RMSE (g)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
