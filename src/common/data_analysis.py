import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_df['label'].value_counts().plot(kind='bar')
plt.title("Distribution of labels")
#plt.show()

##Feature Distribution##

feature_susbet = [f'x{i}' for i in range(20)]

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15,12))
axes = axes.ravel()

for idx,col in enumerate(feature_susbet):
    axes[idx].hist(train_df[col], bins=30)
    axes[idx].set_title(col)

plt.tight_layout()
plt.show()

##Correlation heatmap##
corr = train_df.corr()
sns.heatmap(corr, cmap="coolwarm")
#plt.show()