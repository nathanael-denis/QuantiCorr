import matplotlib.pyplot as plt

# F1-scores across 10 rounds
f1_test = [0.8477, 0.8421, 0.842, 0.8515, 0.8558, 0.8676, 0.8508, 0.8305, 0.8434, 0.8625]
f1_ood_35_45 = [0.8381, 0.8434, 0.8388, 0.8603, 0.8309, 0.8422, 0.8456, 0.8348, 0.8536, 0.8548]
f1_ood_45_55 = [0.8462, 0.8406, 0.8358, 0.8473, 0.8431, 0.8476, 0.8347, 0.8484, 0.8584, 0.8497]

rounds = list(range(1, 11))

plt.figure(figsize=(8,5))
plt.plot(rounds, f1_test, marker='o', color='royalblue', label='Test [10-35°C]')
plt.plot(rounds, f1_ood_35_45, marker='s', color='orange', label='OOD [35-45°C]')
plt.plot(rounds, f1_ood_45_55, marker='^', color='firebrick', label='OOD [45-55°C]')

plt.xlabel('Round')
plt.ylabel('F1-score')
#plt.title('F1-score across rounds for in-distribution and OOD temperatures')
plt.ylim(0.82, 0.87)
plt.xticks(rounds)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
