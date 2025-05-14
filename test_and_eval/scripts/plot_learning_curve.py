import pandas as pd
import matplotlib.pyplot as plt

# Load the learning curve data
df = pd.read_csv('../results/learning_curve.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['train_fraction'], df['nb_train_acc'], label='Naive Bayes Train', marker='o')
plt.plot(df['train_fraction'], df['nb_test_acc'], label='Naive Bayes Test', marker='o')
plt.plot(df['train_fraction'], df['oner_train_acc'], label='OneR Train', marker='s')
plt.plot(df['train_fraction'], df['oner_test_acc'], label='OneR Test', marker='s')
plt.plot(df['train_fraction'], df['j48_train_acc'], label='J48 Train', marker='^')
plt.plot(df['train_fraction'], df['j48_test_acc'], label='J48 Test', marker='^')

plt.xlabel('Training Set Fraction')
plt.ylabel('Accuracy (%)')
plt.title('Learning Curves: Naive Bayes vs OneR vs J48')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../results/learning_curve.png')
plt.show() 