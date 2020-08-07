import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings

#数据预处理
df_train = pd.read_csv("titanic_train.csv",index_col="PassengerId")
train_label = df_train.Survived.values.tolist()
df_train.drop(['Survived'], axis=1, inplace=True)
df_test = pd.read_csv("test.csv", index_col="PassengerId")
# 合并训练集和测试集，以便统一处理
data = df_train.append(df_test, sort=False)

# 处理缺失值
# Age的缺失值，用插值法填充
data['Age'] = data['Age'].fillna(data['Age'].interpolate())
# Embarked的缺失值用众数填充
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])#众数有多个
embarked2 = pd.get_dummies(data.Embarked, prefix='Embarked')
data = pd.concat([data, embarked2], axis=1)  # 将编码好的数据添加到原数据上
data.drop(['Embarked'], axis=1, inplace=True)  # 过河拆桥
# Fare缺失值用众数填充
data['Fare'] = data['Fare'].fillna(data['Fare'].mode()[0])

# 特征工程
#Sex取值数值化：Female为0，Male为1
data.loc[data['Sex'] == 'male','Sex'] = 1
data.loc[data['Sex'] == 'female','Sex'] = 0
data['Sex'] = data['Sex'].astype('int')

#Age属性转换
#child:(0,6]
#teenager:(7,18]
#youth:(18,40]
#midle-aged:(40,60]
#old:>60
data.loc[data['Age'] <= 6,'Age'] = 0
data.loc[(data['Age'] > 6) & (data['Age'] <= 18),'Age'] = 1
data.loc[(data['Age'] > 18) & (data['Age'] <= 40),'Age'] = 2
data.loc[(data['Age'] > 40) & (data['Age'] <= 60),'Age'] = 3
data.loc[data['Age'] > 60,'Age'] = 4
data['Age'] = data['Age'].astype('int')

#Fare属性转换：
#Low: 0-100
#Medium: 100-200
#High: 200-300
#Top: >300
data.loc[data['Fare'] <= 100,'Fare'] = 0
data.loc[(data['Fare'] > 100) & (data['Fare'] <= 200),'Fare'] = 1
data.loc[(data['Fare'] > 200) & (data['Fare'] <= 300), 'Fare'] = 2
data.loc[data['Fare'] > 300,'Fare'] = 3
data['Fare'] = data['Fare'].astype('int')

# Name信息提取
titleDict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}
data.Name = data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
data.Name = data.Name.map(titleDict)
title = pd.get_dummies(data.Name, prefix='title')
data = pd.concat([data, title], axis=1)
data = data.drop(['Name', 'Ticket'], axis=1)

# 方法2：将船舱提取出来，剩下的填充"U"
data.Cabin = data.Cabin.fillna('U')  # 先填充，否则apply无法处理N/A
data.Cabin = data.Cabin.apply(lambda x: x[0])
cabin2 = pd.get_dummies(data.Cabin, prefix="Cabin")
data = pd.concat([data, cabin2], axis=1)
data.drop(['Cabin'], axis=1, inplace=True)

# 获取特征向量
vecs = []
for index, row in data.iterrows():
    vecs.append(list(row))

# 对数据进行标准化和归一化
from sklearn.preprocessing import StandardScaler,MinMaxScaler
vecs = np.array(vecs)
sscl = StandardScaler()
mscl = MinMaxScaler()
vecs=sscl.fit_transform(vecs)
vecs=mscl.fit_transform(vecs)

#划分训练集和测试集
train_data = vecs[0:891]
test_data = vecs[891:]

# 自动调参
def LR_optimization(train_data,train_label):
    warnings.filterwarnings("ignore")
    # 参数设置
    params = {'C':[0.001, 0.01, 0.1, 1, 10],
              "penalty":["l1","l2"],
              'max_iter':[50, 100, 200, 500],
              'solver':['liblinear','sag','lbfgs','newton-cg']
             }
    lr = LogisticRegression()
    clf = GridSearchCV(lr, param_grid=params,scoring="accuracy", cv=10)
    clf.fit(train_data,train_label)
    print("Best score: %0.3f" % clf.best_score_)
    print("Best parameters set:",clf.best_params_)

#LR_optimization(train_data,train_label)

# 使用最优参数进行建模预测
LR_clf = LogisticRegression(C=1, penalty="l1", max_iter=50, solver="liblinear")
LR_clf.fit(train_data,train_label)
id = df_test.index.values.tolist()
pred = LR_clf.predict(test_data)
pred_df = pd.DataFrame({"PassengerId":id, "Survived": pred})
pred_df.to_csv("pred_LR_3.csv", header=True, index=None)
LR_scores = cross_val_score(LR_clf, train_data,train_label, cv=10)
print("LR",LR_scores.mean())