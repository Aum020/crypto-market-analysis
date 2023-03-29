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



#import seaborn as sns
import matplotlib.pyplot as plt


st.title('Crypto Currency Web Application')


image = Image.open("crypto_img.jpg")
st.image(image, use_column_width=True)
# Create a sidebar header
st.sidebar.header('User Input')

key = "placeholder"

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2020-03-23")
    end_date = st.sidebar.text_input("End Input", "2023-03-16")
    crypto_symbol = st.sidebar.text_input("Stock Symbol", "BTC-USD")
    return start_date, end_date, crypto_symbol



start , end ,symbol = get_input()

#crypto_curr = ['BNB-USD','BTC-USD', 'ETH-USD', 'XRP-USD']


data = yf.download(symbol,start,end)

tickerData = yf.Ticker(symbol)

tickerDf = tickerData.history(period='1d',start=start,end=end)


st.subheader('Crypto Data')
st.write(data.head())

st.subheader("Opening Data")
st.line_chart(tickerDf.Open)


st.subheader("Closing Data")
st.line_chart(tickerDf.Close)

st.subheader("Volume Data")
st.line_chart(tickerDf.Volume)





#ML model 



#historical_prices = tickerData.history(period="max")

#closing_prices = historical_prices['Close']


n_cols = 1
dataset = data['Close']
dataset = pd.DataFrame(dataset)
data = dataset





scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(np.array(data))


train_size = int(len(data) * 0.75)
test_size = len(data) - train_size



train_data = scaled_data[0:train_size, :]
#train_data.shape

x_train = []
y_train = []
time_steps = 60
n_cols = 1

for i in range(time_steps , len(train_data)) :
    x_train.append(train_data[i-time_steps:i, :n_cols])
    y_train.append(train_data[i , :n_cols])

        
        
x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],n_cols))



model = Sequential([
    LSTM(50,return_sequences= True, input_shape = (x_train.shape[1],n_cols)),
    LSTM(64,return_sequences= False),
    Dense(32),
    Dense(16),
    Dense(n_cols)
])



model.compile(optimizer= 'adam',loss ='mse',metrics='mean_absolute_error')





history  = model.fit(x_train,y_train,epochs= 30 , batch_size=32)


def plot_losses(history):
    fig = plt.figure(figsize = (12,8))
    plt.plot(history.history["loss"])
    plt.plot(history.history["mean_absolute_error"])
    plt.legend(['Mean Squared Error','Mean Absolute Error'])
    plt.title("Losses")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    #plt.show()
    png_image = fig_to_png(fig)
    st.image(png_image)
    
    
def fig_to_png(fig):
    """
    Converts a Matplotlib figure to a PNG image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf



# display the image in Streamlit


# define a helper function to convert Matplotlib figure to PNG image
plot_losses(history)

time_steps = 60
test_data = scaled_data[train_size - time_steps:, :]

x_test = []
y_test = []
n_cols = 1



for i in range(time_steps , len(test_data)):
    x_test.append(test_data[i-time_steps:i , 0:n_cols])
    y_test.append(test_data[i,0:n_cols])


    
x_test, y_test = np.array(x_test) , np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],n_cols))




predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)
#predictions.shape

y_test = scaler.inverse_transform(y_test)

rmse = np.sqrt(np.mean(y_test - predictions)**2).round(2)
#rmse


preds_acts = pd.DataFrame(data ={'Predictions': predictions.flatten(), 'Actuals':y_test.flatten()})


train = dataset.iloc[:train_size, :1]
test =  dataset.iloc[train_size: , :1]
test['Predictions'] = predictions




fig = plt.figure(figsize=(16, 6))
plt.title('Coin price predictions', fontsize=20)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close', fontsize=18)
plt.plot(train['Close'], linewidth=3)
plt.plot(test['Close'], linewidth=3)
plt.plot(test['Predictions'], linewidth=3)
plt.legend(['Train', 'Test', 'Predictions'])

# Display the plot in Streamlit using st.pyplot()
st.pyplot(fig)

