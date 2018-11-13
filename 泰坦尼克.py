# -*- coding: utf-8 -*-

import pandas as pd

train = pd.read_csv("titanic_train.csv")
test = pd.read_csv("test.csv")

print(train.info())
print("--"*40)
print(test.info())
print("--"*40)

selectd_features = ['Pclass','Sex', 'Age', 'Embarked','SibSp','Parch', 'Fare']

X_train = train[selectd_features]
X_test = test[selectd_features]

y_train = train['Survived']

#通过之前对数据的总体观察，得知Embarked特征存在缺失值，需要补充
print (X_train['Embarked'].value_counts())
print("--"*40)
print (X_test['Embarked'].value_counts())
print("--"*40)

#对于Embarked这种类别的型的特征，我们使用出现频率最高的特征值来填充，
#这也是相对可以减少引入误差的一种填充方法

X_train['Embarked'].fillna('S', inplace = True)
X_test['Embarked'].fillna('S', inplace = True)

#对于Age这种数值类型的特征，我们习惯使用求平均值或者中位数来填充缺失值，
#也是相对可以减少引入误差的一种填充方法

X_train['Age'].fillna(X_train['Age'].mean(), inplace = True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace = True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace = True)

#重新处理后的训练和测试数据进行验证
print(X_train.info())
print("--"*40)
print(X_test.info())
print("--"*40)

from sklearn.feature_extraction import DictVectorizer

dict_vec = DictVectorizer(sparse = False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient = 'record'))
print(dict_vec.feature_names_)
X_test = dict_vec.fit_transform(X_test.to_dict(orient = 'record'))

from sklearn.ensemble import RandomForestClassifier
#使用默认配置初始化RandomForestClassifier
rfc = RandomForestClassifier()

#从流行的工具包XGBoost导入XGBClassifier
from xgboost import XGBClassifier
xgbc = XGBClassifier()


from sklearn.cross_validation import cross_val_score

#使用5折交叉验证的方法在训练集上分别对默认配置的RandomForestClassifier和
#XGBClassifier进行性能评估，并获得平均分类器准确的得分

cross_val_score(rfc, X_train, y_train, cv= 5).mean()

cross_val_score(xgbc, X_train, y_train, cv= 5).mean()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
cross_val_score(lr, X_train, y_train, cv= 5).mean()

#使用默认配置的RandomForestClassifier进行预操作
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 
                               'Survived':rfc_y_predict})
#将RandomForestClassifier测试数据存储在文件中
rfc_submission.to_csv('rfc_submission.csv',index = False)

xgbc.fit(X_train, y_train)

xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 
                               'Survived':xgbc_y_predict})
#将RandomForestClassifier测试数据存储在文件中
xgbc_submission.to_csv('xgbc_submission.csv'
                       ,index = False)

#使用并行网格搜索的方式寻找更好的超参数组合，以期待进一步提供XGBClassifier的预测性能
from sklearn.grid_search import GridSearchCV
params = {'max_depth':list(range(2,7)),'n_estimators':list(range(100,1100,200)),
          'learning_rate':[0.05,0.1,0.25,0.5,1.0]}

xgbc_best = XGBClassifier()
#n_jobs= -1使用计算机全部的CPU核数
gs = GridSearchCV(xgbc_best, params, n_jobs= -1, cv = 5,verbose = 1)
gs.fit(X_train, y_train)


#使用经过优化超参数配置的XGBClassifier的超参数配置以及交叉验证的准确性
print (gs.best_score_)
print (gs.best_params_)

#使用经过优化的超参数配置的XGBClassifier对测试数据的预测结果存储在文件xgbc_best_submission中
xgbc_best_y_predict = gs.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 
                               'Survived':rfc_y_predict})
xgbc_best_submission.to_csv('gbc_submission.csv' ,index = False)


