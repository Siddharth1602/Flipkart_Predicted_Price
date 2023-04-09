from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st



df = pd.read_csv("laptop_details.csv")

df['MRP'] = df['MRP'].str.replace('₹', '')

df['MRP'] = df['MRP'].str.replace(',', '')

df["MRP"] = df["MRP"].astype(int)

## This has been done with the help of regular expression.

df['processor'] = df['Feature'].str.extract('(Intel Core i\d)') # Extracts the processor information
df['ram'] = df['Feature'].str.extract('(\d+ GB DDR4)') # Extracts the RAM information
df['os'] = df['Feature'].str.extract('(Windows \d+)') # Extracts the operating system information
df['ssd'] = df['Feature'].str.extract('(\d+ GB SSD)') # Extracts the SSD information
df['screen_size'] = df['Feature'].str.extract('(\d{2}\.\d+ inch)') # Extracts the screen size information
df['Brand'] = df['Product'].str.extract('([A-Za-z]+)')

df_sorted = df.sort_values('MRP')

df_sorted = df_sorted.drop('Product', axis=1)

df_sorted = df_sorted.drop("Feature",axis = 1)

## Replacing the null values.

df_sorted["Rating"].fillna(df_sorted['Rating'].mean(),inplace = True)
df_sorted["processor"].fillna("Intel Core i5",inplace = True)
df_sorted["ram"].fillna("8 GB DDR4",inplace = True)
df_sorted["os"].fillna("Windows 11",inplace = True)
df_sorted["ssd"].fillna("512 GB SSD",inplace = True)
df_sorted["screen_size"].fillna("15.6 inch",inplace = True)

df_sorted = df_sorted.rename(columns={
    'processor': 'Processor',
    'ram': 'RAM',
    'os': 'Operating System',
    'ssd': 'Storage',
    'screen_size': 'Screen Size',
})
df_sorted['Operating System'] = df_sorted['Operating System'].str.replace('Windows', '')
df_sorted['Storage'] = df_sorted['Storage'].str.replace('GB SSD', '')
df_sorted['Screen Size'] = df_sorted['Screen Size'].str.replace('inch', '')
df_sorted['RAM'] = df_sorted['RAM'].str.replace('GB DDR4', '')

# select the features and the target variable
X = df_sorted[['RAM', 'Storage', 'Operating System', 'Screen Size']]
y = df_sorted['MRP']


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
## Creating the Interface:

st.set_page_config(page_title= "Laptop Price Prediction",
                   page_icon= ":bar_chart",layout = "wide")
st.title("Laptop Price Predictor 💻")


# making 2 cols left_column, middle_column, right_column
left_column, right_column = st.columns(2)
with left_column:
    # RAM input
    RAM = st.selectbox("RAM (in_GB)", df_sorted["RAM"].unique())

with right_column:
    # Storage input
    Storage = st.selectbox("Storage (SSD)", df_sorted["Storage"].unique())

left_column, right_column = st.columns(2)
with left_column:
    operating_system = st.selectbox("Operating System (Windows)",
                                    df_sorted["Operating System"].unique())
    
with right_column:
    screen_size = st.selectbox("Screen Size (in_inches)",
                               df_sorted["Screen Size"].unique())
    
def predict_price(RAM, Storage, operating_system, screen_size):
    # create a new dataframe with the input values
    new_laptop = pd.DataFrame({'RAM': [RAM], 'Storage': [Storage], 'Operating System': [operating_system], 'Screen Size': [screen_size]})

    
    # make the prediction
    predicted_price = model.predict(new_laptop)[0]
    
    return predicted_price

# add a button to get the predicted price
if st.button("Get Predicted Price"):
    # call the predict_price function with the input values
    predicted_price = predict_price(RAM, Storage, operating_system, screen_size)
    
    # display the predicted price
    st.write("The predicted price for the laptop is", round(predicted_price, 2), "rupees.")
