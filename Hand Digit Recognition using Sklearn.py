


'''@ Author Vishnu Varthan
Application of  diffferent algorithms using toy dataset

'''

from sklearn.datasets import  load_digits

digits=load_digits()
#Test casee please uncomment if u need it so
''''print(iris.data[0])
print(digits.images[0])'''
'''Classifying handwritten digits in the  toy data set using Support Vector machine algorithm which is a more efficient way for 
classification problems'''

from sklearn import svm

clf=svm.SVC(gamma=0.1,C=100)

'''C represents how mucb we penalize the error
If C is larger it has low bias and high variance and vice versa and C represents the softline of the SVM  which ignores 
some data points


'''
training_set,test_set=digits.data[:-1],digits.data[-1:]

training_feature,test_feature=digits.target[:-1],digits.target[-1:]

#Magic happens
clf.fit(training_set,training_feature)
clf.predict(test_set)




