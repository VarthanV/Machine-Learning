
# coding: utf-8

# ## Fun stuffs

# In[18]:


from sklearn.neighbors  import KNeighborsClassifier
#Prerequistite
from sklearn.feature_extraction.text  import CountVectorizer
import numpy as np
import pandas as pd
lang_of_girls=['mmmmmm i hate u a lot(:(:','kkkkk i hate hate u','bye imm gonna block you''']

'''Learn the vocabulary of girls ,pretty hard though(:'''
vector=CountVectorizer()
vector.fit(lang_of_girls)

document_mat=vector.transform(lang_of_girls)
'''Transform the vector'''
array=document_mat.toarray()
print(vector.get_feature_names())
file=pd.DataFrame(array,columns=vector.get_feature_names())
test=['I hate u a lot']
test_dtm=vector.transform(test)


d= pd.DataFrame(test_dtm.toarray(),columns=vector.get_feature_names())



    


# ## The actual stuffs

# In[19]:


import pandas as pd
file='sms.tsv'
sms=pd.read_table(file,header=None,names=['label','message'])


# In[20]:


#Examine the class distribution
sms.label.values


#   ## Finding how many values are there in the both classes

# In[21]:



sms.label.value_counts()


# ## Add a new column that maps ham to 0 and spam to 1 for easy computation

# In[22]:


sms['mapped']=sms.label.map({'ham':0,'spam':1})


# In[23]:


sms.head(10)


# ## Training the model

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
vectorizer=CountVectorizer()







# ## Equally combining fit and transform
# 
# 
# 
# 
# 

# In[31]:


dtm=vectorizer.fit_transform(x_train)


# In[32]:


dtm.shape


# ## In this 4179 represents the number of rows (ie ,words) and 7456 represents what the model has learnt (i.e) induvidual words etc..

# ## Using Naive Bayes Classification model
# 
# 

# In[ ]:




