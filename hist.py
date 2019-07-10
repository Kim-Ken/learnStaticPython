from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import scipy

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.loc[df['target'] == 0, 'target'] = "setosa"
df.loc[df['target'] == 1, 'target'] = "versicolor"
df.loc[df['target'] == 2, 'target'] = "virginica"

a=iris.data[:-1,2]
b=iris.data[:-1,3]

"""print("データ:a")
print(a)
print("a:平均")
print(np.mean(a))
print("b:平均")
print(np.mean(b))
"""
label = [
    '20 - 24',
    '25 - 29',
    '30 - 34',
    '35 - 39',
    '40 - 44',
    '45 - 49',
    '50 - 54',
    '55 - 59'
]

# 描画
plt.show()
#print(type(a))

"""plot.plot(a,b,'x')
sns.pairplot(df, hue="target")
#print(np.corrcoef(a,b))
plot.show()"""
