import re
import matplotlib.pyplot as plt
import numpy as np

# Paths to result files (before and after improvement)
algorithms = ['Naive Bayes', 'OneR', 'J48', 'Association Rules']
tags = ['before', 'after']
file_template = 'results/reports/{}/{}_eval.txt'
file_map = {
    'Naive Bayes': 'naive_bayes',
    'OneR': 'oneR',
    'J48': 'j48',
    'Association Rules': 'association_rules',
}

accuracies = {alg: {tag: None for tag in tags} for alg in algorithms}
for alg in algorithms:
    for tag in tags:
        path = file_template.format(tag, file_map[alg])
        try:
            with open(path, 'r') as f:
                content = f.read()
                match = re.search(r'Accuracy:\s*([0-9.]+)', content)
                if match:
                    accuracies[alg][tag] = float(match.group(1))
        except Exception as e:
            accuracies[alg][tag] = None
            print(f"Could not read {path}: {e}")

# Prepare data for plotting
x = np.arange(len(algorithms))
width = 0.35
before_vals = [accuracies[alg]['before'] if accuracies[alg]['before'] is not None else 0 for alg in algorithms]
after_vals = [accuracies[alg]['after'] if accuracies[alg]['after'] is not None else 0 for alg in algorithms]

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, before_vals, width, label='Before Improvement', color='#4e79a7')
rects2 = ax.bar(x + width/2, after_vals, width, label='After Improvement', color='#f28e2b')

# Add value labels
for i, (b, a) in enumerate(zip(before_vals, after_vals)):
    if accuracies[algorithms[i]]['before'] is not None:
        ax.text(x[i] - width/2, b + 1, f'{b:.2f}%', ha='center', va='bottom', fontsize=12)
    else:
        ax.text(x[i] - width/2, 1, 'N/A', ha='center', va='bottom', fontsize=12, color='#e15759')
    if accuracies[algorithms[i]]['after'] is not None:
        ax.text(x[i] + width/2, a + 1, f'{a:.2f}%', ha='center', va='bottom', fontsize=12)
    else:
        ax.text(x[i] + width/2, 1, 'N/A', ha='center', va='bottom', fontsize=12, color='#e15759')

# Annotate Association Rules
ax.annotate('Association Rules is not a classifier\n(no accuracy metric)',
            xy=(3, 0), xycoords='data', xytext=(2.5, 30), textcoords='data',
            arrowprops=dict(facecolor='#e15759', shrink=0.05),
            fontsize=12, color='#e15759', ha='center')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Algorithm Performance Before and After Improvement')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.savefig('results/figures/classifier_accuracies_before_after.png')
plt.show()

# Placeholder for future: plot confusion matrix heatmaps, etc.
# You can extend this script to parse and visualize confusion matrices from the result files. 