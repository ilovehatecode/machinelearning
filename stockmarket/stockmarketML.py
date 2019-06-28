import pandas as pd
import numpy as np
import os

df_SandP = pd.read_csv(os.getcwd()+'/data/SandPquarterlyrate.csv')
df_numpy_SandP = df_SandP.to_numpy()
df_SandP_ = df_numpy_SandP.tolist


features_list = np.reshape(np.array([-4.4,-0.6,1.5,4.5,1.5,3.7,3.0,2.0,-1.0,2.9,-0.1,4.7,3.2,1.7,0.5,0.5,3.6,0.5,3.2,3.2,-1.0,5.1,4.9,1.9,3.3,3.3,1.0,0.4,1.5,2.3,1.9,1.8,1.8,3.0,2.8,2.3,2.2,4.2,3.4,2.2,3.1
]), (-1,1))
target_list = []
for element in df_numpy_SandP:
    target_list.append(element[1])

target_list = np.reshape(np.array(target_list)[:41], (-1,1))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features_list, target_list, test_size=.3, random_state=42)

reg = LinearRegression()
##Train Linear Regression
reg = reg.fit(feature_train, target_train)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(feature_test, target_test))


train_color = "b"
test_color = "r"

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
