#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
plt.style.use ("dark_background")

import os


# In[3]:


from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version =1, return_X_y=True)


# In[4]:


X.shape


# # 80% : 20%

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size =2/7, random_state=0)


# In[6]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[7]:


type(X_train)


# In[8]:


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
#reshape and scale to be in [0,1]
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[ ]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


plt.figure(figsize=(20, 4))
for index in range(5):
    plt.subplot(1,5, index+1)
    plt.imshow(X_train[index].reshape((28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % int(y_train.to_numpy()[index]), fontsize=20)


# In[ ]:


#Make an instance of the Model
#mlp = MLPClassifier()
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50),activation="relu", solver= 'adam', max_iter=200)


# In[ ]:


mlp.fit(X_train_scaled, y_train)


# In[32]:


#predictions
predictions = mlp.predict(X_test_scaled)


# In[35]:


#evaluation
score = mlp.score(X_test_scaled, y_test)
print(score)


# In[9]:


total_params_mlp = (100 * 50) + 50 + (50 * 1) + 1
print("Total number of trainable parameters in the MLP model:", total_params_mlp)


# In[10]:


logistic_regression_params = (100 * 1) + 1
print("Total number of trainable parameters in the logistic regression model:", logistic_regression_params)


# In[36]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=mlp.classes_.tolist()))


# In[40]:


cm = cm(y_test, predictions)


# In[41]:


cm


# In[42]:



plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".2f", linewidth=0.5, square=True, cmap="Blues_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title('Accuracy Score: {0}'.format(score), size=15)
plt.show()


# In[44]:


plt.plot(mlp.loss_curve_)
plt.title("Loss Curve", fontsize=18)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


# # BAGGING ALGORITHM ON 80:20%

# In[45]:


#bagging algorithm
from sklearn.ensemble import BaggingClassifier
BaggingClassifier


# In[46]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
bag = BaggingClassifier(knn, max_samples =.5, max_features=28, n_estimators=20)


# In[47]:


bag.fit(X_train, y_train)


# In[48]:


y_pred = bag.predict(X_test)
accuracy_score(y_test, y_pred)


# In[49]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=bag.classes_.tolist()))


# In[50]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
predictions = bag.predict(X_test)
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=bag.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Iris Dataset")
plt.show()


# # random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('pinfo', 'RandomForestClassifier')


# In[52]:


rf = RandomForestClassifier(n_estimators=20)


# In[53]:


rf.fit(X_train, y_train)


# In[54]:


#evaluae the model
y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)


#  # adaboost

# In[55]:


from sklearn.ensemble import AdaBoostClassifier
get_ipython().run_line_magic('pinfo', 'AdaBoostClassifier')


# In[59]:


ada=AdaBoostClassifier(n_estimators=50)


# In[60]:


ada.fit(X_train, y_train)


# In[61]:


#evaluae the model
y_pred = ada.predict(X_test)
accuracy_score(y_test, y_pred)


# # knn

# In[62]:


from sklearn.neighbors import KNeighborsClassifier
get_ipython().run_line_magic('pinfo', 'KNeighborsClassifier')


# In[63]:


knn=KNeighborsClassifier(n_neighbors=3)


# In[64]:


knn.fit(X_train, y_train)


# In[65]:


#evaluate the model
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:




