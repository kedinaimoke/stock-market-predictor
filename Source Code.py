#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf


# In[2]:


sp500 = yf.Ticker("^GSPC")


# In[3]:


sp500 = sp500.history(period="max")
# querying all trading days recorded in S & P 500 history


# In[4]:


sp500


# In[5]:


sp500.index
# this is to call the datetime index of stocks in a dictionary format to enable indexing and slicing of dataframe


# In[6]:


# next comes cleaning up and visualising stock market data
sp500.plot.line(y="Close", use_index=True)
# plotting the closing price (y) against the index (x)


# In[7]:


# removing of dividend and stock splits columns as they are more approrpriate for individual stocks and not large data
del sp500["Dividends"]
del sp500["Stock Splits"]


# In[8]:


# moving on to setting up the target (for the model)
sp500["Tomorrow"] = sp500["Close"].shift(-1)


# In[9]:


sp500


# In[10]:


# now, there is a column showing tomorrow's price and based on the price, a target will be set.
# this will be done by checking if tomorrow's price is greater than today's price.
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


# In[11]:


sp500
# calling the index to get 1 for when prices were up and 0 for when they went low.


# In[12]:


# Going too far in market price (years) can affect the prediction of our model, so we will be working with data from 1990
sp500 = sp500.loc["1990-01-01":].copy()
# .copy is essential for when dataframes need to be a subset else an error will occur.


# In[13]:


sp500


# In[14]:


# moving on training the model
from sklearn.ensemble import RandomForestClassifier

# RandomForest is used as the model because it trains several individual decision trees with randomised parameters and 
# averaging the results. This makes them more resistant to "overfit" (not making accurate predictions on data) compared
# to other models. They also run relatively quicker and are better at picking up non-linear tendencies in the data.
# Currently, the open price has no linear relationship with the target price.

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
# n_estimators is the number of decision trees. The higher the decision trees, the better the accuracy.
# min_samples_split helps protect against overfit. The higher it is, the less the accuracy but the better the model
# is protected against overfit.
# random_state sets a randomnisation set to the model so that if it is ran twice, the random numbers will be in a 
# predictable sequence.

train = sp500.iloc[:-100]
# data is split into train and test set because cross set validation is not used. This is because the results of cross
# set validation will look excellent as a model but not in the real world because it does not take in the time series
# of the data which results in leakage.
# This will be done by putting all the rows except the last 100 in the training set and the last 100 rows into the
# test set.

test = sp500.iloc[-100:]

# next, a predictors list is created with all the columns (apart from the Tomorrow & Target columns) to predict the target.
predictors = ["Close", "Volume", "Open", "High", "Low"]

# now to fit the model by training it with the predictors variable and target column
model.fit(train[predictors], train["Target"])


# In[15]:


from sklearn.metrics import precision_score

# what precision_score does is that ensures that target either 0 or 1 was actually 0 or 1 for the concerned day of trade
preds = model.predict(test[predictors])
# preds varibale generates predictions


# In[16]:


import pandas as pd

preds = pd.Series(preds, index=test.index)
# This imports the predictions in a series format other than as an array with zeroes or ones that have no dates


# In[17]:


preds


# In[18]:


precision_score(test["Target"], preds)
# This is to calculate the actual target and predicted target in the precision score which returns a "no-so-great"
# precision score (37%)


# In[19]:


# next, to make the model better, the predictions are done by combining predicted values with actual values
combined = pd.concat([test["Target"], preds], axis=1)
# axis=1 is so that each input is treated as a column


# In[20]:


combined.plot()
# this is to plot the predicted values with the actual values


# In[21]:


# now, buidling a backtesting system as a more robust means of testing
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[22]:


def backtest(data, model, predictors, start=2500, step=250):
#     start=2500 is saying "take 10 years of data and train my model with 10 years of data"
#     step=250 is saying "the model should be trained for each year"
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)
# to concatenate all predictions together for backtesting


# In[23]:


predictions = backtest(sp500, model, predictors)


# In[24]:


predictions["Predictions"].value_counts()
# to check how many days the prices were predicted to go up versus the days the prices would be down


# In[25]:


# now working with precision score which is proposed to be higher
precision_score(predictions["Target"], predictions["Predictions"])


# In[26]:


predictions["Target"].value_counts() / predictions.shape[0]
# this is to check the percentage of days the value actually went up


# In[27]:


# now to add more predictors the model to make it more accurate
# A typical human would use several days of trades to predict future prices, however with model this can be done
# by getting the mean close price of the past 2, 5, 60, 250 and 1000 trading days
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
#     Around the loop, the result with the ratio of today's price is the past 2 trading days, 5 trading days and so on
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
#     This checks any given trading day and gets the average sum of the Target for the past trading days

    new_predictors += [ratio_column, trend_column]


# In[28]:


sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])
# this is to drop rows that failed to produce enough data


# In[29]:


sp500


# In[30]:


# now, to improve the model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[31]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1] # this returns the probability of the stock price
    preds[preds >= .6] = 1 # this sets a custom prediction at 60%
    preds[preds < .6] = 0 # this reduces the total number of training days that the price will go up and
    # increase the chance that the price will go up
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[32]:


# performing backtesting again
predictions = backtest(sp500, model, new_predictors)


# In[33]:


predictions["Predictions"].value_counts()
# to check number of good trading days


# In[34]:


precision_score(predictions["Target"], predictions["Predictions"])
# to check accuracy of predicted trading days


# In[35]:


sp500 = pd.read_csv("sp500.csv", index_col=0)
sp500.to_csv("sp500.csv")
# dataset file.


# In[36]:


sp500.index = pd.to_datetime(sp500.index)


# In[37]:


sp500


# In[38]:


predictions

