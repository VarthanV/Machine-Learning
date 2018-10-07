# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 04:54:21 2018

@author: Vishnu Varthan
"""
import numpy as np
a=np.array([[1,2,3],[4,5,6]])
print(a)
# Seeding the random as it gives the same value next time
np.random.seed(123)
b=np.random.randn(4)
print(b)
diagonal=np.diag(b)
print(diagonal)
import matplotlib.pyplot as plt
x=np.linspace(0,10,10)
y=np.linspace(0,30,10)
plt.plot(x,y,'r-')
plt.show()
print(diagonal[2,2:3])
repeat=np.tile(a,6)
print(repeat)
prime=np.ones((100,),dtype=bool)

prime[:2]=0
print(prime)
'''Fancy indexing'''
fancy=np.random.randint(1,100,5)
''' We create a mask for the purpose of filtering of array based on our neeeds'''
mask=(fancy%3==0)

