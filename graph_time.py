import matplotlib.pyplot as plt
import numpy as np

methods = ['Original', 'PCA', 't-SNE', 'LLE']
times = [28.6756, 22.1937, 68.8366, 42.4167]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(methods, times, color=['blue', 'green', 'red', 'orange'])

ax.set_ylabel('Processing Time (seconds)')
ax.set_title('Data Processing Time Comparison')
ax.set_ylim(0, max(times) * 1.1)  # Set y-axis limit with some padding

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s',
            ha='center', va='bottom')
    
plt.tight_layout()
plt.show()