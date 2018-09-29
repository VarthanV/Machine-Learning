
# coding: utf-8

# In[46]:


'''Classifying Cancers Malignant of benign using Sci-Kit learn
@ Author Vishnu Varthan,inspired by the design of Vicky Raj in Keras


'''
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


cancer=load_breast_cancer()
data=cancer.data
target=cancer.target
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.50,random_state=4)
clf=KNeighborsClassifier(n_neighbors=11)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(accuracy_score(y_test,y_pred))






















# In[22]:


cd Desktop


