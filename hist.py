from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    '1.0 - 2.0',
    '2.0 - 3.0',
    '3.0 - 4.0',
    '4.0 - 5.0',
    '5.0 - 6.0',
    '6.0 -'
]

fig, ax = plt.subplots()

# 8個の階級でヒストグラムを作成します。binsの最小値と最大値をrangeで指定します。
# 戻り値について　n => 各階級における度数、bins => 階級のリスト
n, bins, patches  = ax.hist(a, bins=6, range=(1.0, 7.0))

ax.set_xticks([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
# X軸にラベルをセット、90度回転させる
ax.set_xticklabels(label, rotation = 90)

# 描画
plt.show()
#print(type(a))

"""plot.plot(a,b,'x')
sns.pairplot(df, hue="target")
#print(np.corrcoef(a,b))
plot.show()"""
