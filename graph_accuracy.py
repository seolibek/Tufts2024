import matplotlib.pyplot as plt

# Data
methods = ['Original', 'PCA', 't-SNE', 'LLE']
k_values = list(range(2, 9))
accuracies = {
    'Original': [23.48, 0, 0, 21.36, 0, 0, 0],
    'PCA': [0, 21.36, 21.36, 21.36, 8.55, 0, 0],
    't-SNE': [24.67, 24.49, 23.30, 0, 7.76, 21.36, 7.76],
    'LLE': [0, 25.08, 0, 0, 0, 0, 0]
}

# Create the plot
plt.figure(figsize=(12, 6))

for method in methods:
    plt.plot(k_values, accuracies[method], marker='o', label=method)

plt.xlabel('K Value')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs K Value for Different Data Reduction Methods')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Set x-axis ticks to integer values
plt.xticks(k_values)

# Set y-axis range from 0 to the maximum accuracy value (rounded up to nearest 5)
max_accuracy = max(max(acc) for acc in accuracies.values())
plt.ylim(0, (max_accuracy // 5 + 1) * 5)

plt.tight_layout()
plt.show()