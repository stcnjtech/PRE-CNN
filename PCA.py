import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_excel('Data01.xls')
dt = pd.read_csv('data.csv')
dx = pd.read_csv('X_new.csv')

def norm_(x):
    xmean = np.mean(x,0)
    std = np.std(x,0)
    return (x-xmean)/std

data_ = norm_(df)
data_.drop(columns=['Gene_ID','Gene_Name','Promoterintensity','Promoter sequence','cluster'],inplace=True)

ew, ev = np.linalg.eig(np.cov(data_.T))
ew_order = np.argsort(ew)[::-1]
ew_sort = ew[ew_order]
ev_sort = ev[:,ew_order]
V = ev_sort[:,:2]
X_new = data_.dot(V)


# ================Scatter图================
Sc = plt.scatter(X_new.iloc[:,0],X_new.iloc[:,1],marker='+',s=100,c=dt.cluster,cmap=plt.cm.coolwarm,edgecolors='face')
plt.colorbar(Sc)
plt.title("Principal Component Analysis", fontsize=12)
plt.xlabel('Principal Component 0')
plt.ylabel('Principal Component 1')
plt.show()
pd.DataFrame(ew_sort).plot(kind='bar')
#plt.show()
print(V)
print(data_.columns)

# ================等高线图================
sns.jointplot(x='Principal Component 0', y='Principal Component 1',data= dx,
              kind="kde",
              hue='Intensity Group',
              joint_kws=dict(alpha = 0.85),
              marginal_kws=dict(shade=True)
)

# ================正负条形图===============

du = pd.read_csv('test.csv')
x = du['PrincipalComponent'].values
a = du['PC0'].values
b = du['PC1'].values
plt.figure(figsize=(16,5))
plt.subplot(1, 1, 1)
plt.bar(x, a, label = 'Principal Component 0')
plt.bar(x, b, label = 'Principal Component 1')
plt.title('Principal Component Proportion')
plt.tick_params(labelsize=8)
plt.legend()
plt.show()
