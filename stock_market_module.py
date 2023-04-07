import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf 
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense , LSTM , Dropout
import io
from datetime import timedelta
import datetime 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.title('Stock Market Web Application')


image = Image.open("stock_image.jpg")
st.image(image, use_column_width=True)

st.sidebar.header('User Input')

key = "placeholder"


def get_input():
    start_date = st.sidebar.text_input("Start Date", "2020-03-23")
    end_date = st.sidebar.text_input("End Input", "2023-03-16")
    stock_symbol = st.sidebar.text_input("Stock Symbol", "TATAMOTORS.NS")
    return start_date, end_date, stock_symbol


start , end ,symbol = get_input()


data = yf.download(symbol,start,end)

data = data.reset_index()

tickerData = yf.Ticker(symbol)

tickerDf = tickerData.history(period='1d',start=start,end=end)


st.subheader('Stock Data')
st.write(data.head())

st.subheader("Opening Data")
st.line_chart(tickerDf.Open)


st.subheader("Closing Data")
st.line_chart(tickerDf.Close)

st.subheader("Volume Data")
st.line_chart(tickerDf.Volume)


data['Close'] = pd.to_numeric(data.Close,errors='coerce')
data = data.dropna()
trainData = data.iloc[:,4:5].values

sc = MinMaxScaler(feature_range= (0,1))
trainData = sc.fit_transform(trainData)
 

x_train = []
y_train = []

for i in range (60,742):
    x_train.append(trainData[i-60:i,])
    y_train.append(trainData[i,0])
    
x_train,y_train = np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))



model = Sequential()

model.add(LSTM(units=100 , return_sequences=  True,input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))


model.add(LSTM(units=100,return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.compile(optimizer='adam',loss="mean_squared_error")

hist = model.fit(x_train,y_train,epochs=20,batch_size=32,verbose=2)

plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
st.pyplot(plt.gcf())  #graph 


testData =  yf.download(symbol,start,end)
testData["Close"] = pd.to_numeric(testData.Close,errors='coerce')
testData = testData.dropna()
testData = testData.iloc[:,4:5]
y_test = testData.iloc[60:,0:].values

inputClosing = testData.iloc[:,0:].values
inputClosing_scaled = sc.transform(inputClosing)


x_test = []
length = len(testData)
timestep = 60

for i in range(timestep,length):
    x_test.append(inputClosing_scaled[i-timestep:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
 


y_pred = model.predict(x_test)

predicted_price = sc.inverse_transform(y_pred)

#this is hp


fig = plt.figure(figsize=(16,6))
plt.plot(y_test, color = 'red' , label = 'Actual Stock Price')
plt.plot( predicted_price , color = 'green' , label = 'Predicted Stock Price')
plt.title('Stock price prediction',fontsize = 20)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Stock Price' ,fontsize = 18)
plt.legend()
st.pyplot(fig)




