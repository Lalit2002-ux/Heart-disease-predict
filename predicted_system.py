# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 21:11:17 2025

@author: lalit
"""

import numpy as np
import pandas as pd
import pickle

#loaded the save model
Loaded_model=pickle.load(open('L:/logistic ml/logistic_model.sav','rb'))



input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = Loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have asssssss Heart Disease')
else:
  print('The Person has Heart Disease')