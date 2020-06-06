
#import packages
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#1. Take data and parse it
data = pd.read_csv('weight_data.csv')
height_x = data.iloc[:,:-1].values
weight_y = data.iloc[:,1].values

train_height,test_height,train_weight,test_weight = train_test_split(height_x,weight_y,test_size=1/5)

#2. Use parsed data for training
model = LinearRegression()
model.fit(train_height,train_weight)

#3. Do predictions
pred_result = model.predict(test_height)

print("accuracy :",model.score(test_height,test_weight))

print(test_height)
print(test_weight)
print(pred_result)