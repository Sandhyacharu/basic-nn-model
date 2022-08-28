# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Neural Networks Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated.

Regression model A regression model provides a function that describes the relationship between one or more independent variables and a response, dependent, or target variable. For example, the relationship between height and weight may be described by a linear regression mode.

## Neural Network Model
![image](https://user-images.githubusercontent.com/75235167/187084520-1af19950-cff6-4683-81ba-5b2665968baa.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python3
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
import matplotlib.pyplot as plt
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('StudentsData').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float','Output':'float'})
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
df.head()
x=df[['Input']].values
y=df[['Output']].values
x
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=11)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_test1=Scaler.transform(x_test)
x_train1
ai_brain = Sequential([
    Dense(6,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])
ai_brain.compile(
    optimizer='rmsprop',
    loss='mse'
)
ai_brain.fit(x_train1,y_train,epochs=4000)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
plt.title('Training Loss Vs Iteration Plot')
ai_brain.evaluate(x_test1,y_test)
x_n1=[[66]]
x_n1_1=Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```
## Dataset Information

![image](https://user-images.githubusercontent.com/75235167/187084627-f9aa6370-ae23-4a7a-9426-90089b2cf233.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235167/187085030-e7e44921-49f7-4b79-8d45-038802d3c4d0.png)

### Test Data Root Mean Squared Error

0.00301834917627275

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235167/187085090-273cb5b2-9d92-4b02-88c4-e88c9313aaef.png)

## RESULT

Succesfully created and trained a neural network regression model for the given dataset.
