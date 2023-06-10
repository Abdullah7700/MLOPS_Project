#!/usr/bin/env python
# coding: utf-8

# ## Basic Library Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[2]:


from sklearn.preprocessing import LabelEncoder


# In[3]:


import mlflow
import mlflow.sklearn


# ## Data Import and Preprocessing

# In[4]:


df = pd.read_csv('data.csv')
df = df.iloc[0:584524,0:21]
df


# In[5]:


df.columns


# In[6]:


percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df


# In[7]:


df = df.dropna(subset=['status', 'category_name_1', 'Customer Since', 'Customer ID'])
df


# In[8]:


df['Month'] = df['Month'].astype(int)


# In[9]:


df.loc[df['Month'] == 9, 'Month Name'] = 'September' 
df.loc[df['Month'] == 8, 'Month Name'] = 'August' 
df.loc[df['Month'] == 7, 'Month Name'] = 'July' 
df.loc[df['Month'] == 6, 'Month Name'] = 'June' 
df.loc[df['Month'] == 5, 'Month Name'] = 'May' 
df.loc[df['Month'] == 4, 'Month Name'] = 'April' 
df.loc[df['Month'] == 3, 'Month Name'] = 'March' 
df.loc[df['Month'] == 2, 'Month Name'] = 'February' 
df.loc[df['Month'] == 1, 'Month Name'] = 'January' 
df.loc[df['Month'] == 12, 'Month Name'] = 'December' 
df.loc[df['Month'] == 11, 'Month Name'] = 'November' 
df.loc[df['Month'] == 10, 'Month Name'] = 'October'


# In[10]:


df.loc[df['price'] <= 500 , 'Price Range'] = '< 500' 
df.loc[(df['price'] <= 1000) &  (df['price'] > 500), 'Price Range'] = '500 - 1000' 
df.loc[(df['price'] <= 1500) &  (df['price'] > 1000), 'Price Range'] = '1000 - 1500' 
df.loc[(df['price'] <= 2000) &  (df['price'] > 1500), 'Price Range'] = '1500 - 2000' 
df.loc[(df['price'] <= 3000) &  (df['price'] > 2000), 'Price Range'] = '2000 - 3000' 
df.loc[(df['price'] <= 4000) &  (df['price'] > 3000), 'Price Range'] = '3000 - 4000' 
df.loc[(df['price'] <= 5000) &  (df['price'] > 4000), 'Price Range'] = '4000 - 5000' 
df.loc[(df['price'] > 5000), 'Price Range'] = '5000 <' 


# In[11]:


df.loc[df['discount_amount'] <= 500 , 'discount_amount Range'] = '< 500' 
df.loc[(df['discount_amount'] <= 1000) &  (df['discount_amount'] > 500), 'discount_amount Range'] = '500 - 1000' 
df.loc[(df['discount_amount'] <= 1500) &  (df['discount_amount'] > 1000), 'discount_amount Range'] = '1000 - 1500' 
df.loc[(df['discount_amount'] <= 2000) &  (df['discount_amount'] > 1500), 'discount_amount Range'] = '1500 - 2000' 
df.loc[(df['discount_amount'] <= 3000) &  (df['discount_amount'] > 2000), 'discount_amount Range'] = '2000 - 3000' 
df.loc[(df['discount_amount'] > 3000), 'discount_amount Range'] = '3000 <'  


# In[12]:


df = df.drop(['sku', 'sales_commission_code'], axis=1)


# In[13]:


percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df


# In[14]:


df


# ## Category-wise Distribtion & Visualization

# In[14]:


temp = df['category_name_1'].value_counts()
data = pd.DataFrame({'Category':temp.index, 'Count':temp.values})
data


# In[15]:


y = data["Count"].to_list()


# In[16]:


fig = px.bar(data, x="Count", y="Category", orientation='h', text= y)
fig.update_traces(marker_color='pink')
fig.show()


# ## Status-wise Distribtion & Visualization

# In[17]:


df['status'].value_counts()


# In[18]:


temp = df['status'].value_counts()
data = pd.DataFrame({'Status':temp.index, 'Count':temp.values})
data = data.head(7)


# In[19]:


fig = px.pie(data, values="Count", names="Status",
             color_discrete_sequence=px.colors.sequential.RdBu,
             opacity=0.7, hole=0.5)
fig.show()


# ## Payment_Method-wise Distribtion & Visualization

# In[115]:


temp = df['payment_method'].value_counts()
data = pd.DataFrame({'Payment Method':temp.index, 'Count':temp.values})
data


# In[116]:


y = data["Count"].to_list()

fig = px.bar(data, x="Count", y="Payment Method", orientation='h', text= y)
fig.update_traces(marker_color='teal')
fig.show()


# ## BI_Status-wise Distribtion & Visualization

# In[117]:


df['BI Status'].value_counts()
temp = df['BI Status'].value_counts()
data = pd.DataFrame({'Status':temp.index, 'Count':temp.values})
data = data.head(3)


# In[118]:


fig = px.pie(data, values="Count", names="Status", color = "Status", color_discrete_map={'Net':'darkblue',
                                 'Gross':'cyan',
                                 'Valid':'royalblue'}, opacity=0.7, hole=0.5)
fig.show()


# ## Price_Range-wise Distribtion & Visualization

# In[119]:


x = df['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[120]:


x = ["< 500", "500 - 1000", "1000 - 1500", "1500 - 2000", "2000 - 3000", "3000 - 4000", "4000 - 5000 ", "5000 < "]
y = [205146, 136202, 120448, 46009, 28497, 24225, 13744, 10253]


# In[121]:


fig = px.bar(df, x = x, y=y, text= y)
fig.update_traces(marker_color='gold')
fig.show()


# ## Discount_Range-wise Distribtion & Visualization

# In[122]:


x = df['discount_amount Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# ## Price Range Distribuion YoY
# 

# In[123]:


year_2016 = df.loc[df["Year"] == 2016]
year_2017 = df.loc[df["Year"] == 2017]
year_2018 = df.loc[df["Year"] == 2018]


# In[124]:


x = year_2016['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[125]:


x = year_2017['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[126]:


x = year_2018['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# ## Customer-Since Distribution

# In[127]:


df['Customer Since'].value_counts()


# ## Category-Wise Price

# In[128]:


df['category_name_1'].value_counts()


# In[129]:


Mobiles = df.loc[df["category_name_1"] == 'Mobiles & Tablets']
MF = df.loc[df["category_name_1"] == "Men's Fashion"]
WF = df.loc[df["category_name_1"] == "Women's Fashion"]
Apl = df.loc[df["category_name_1"] == 'Appliances']
SS = df.loc[df["category_name_1"] == 'Superstore']
BG = df.loc[df["category_name_1"] == 'Beauty & Grooming']
Sogh = df.loc[df["category_name_1"] == 'Soghaat']
Others = df.loc[df["category_name_1"] == 'Others']
HL = df.loc[df["category_name_1"] == 'Home & Living']
Ent = df.loc[df["category_name_1"] == 'Entertainment']
HS = df.loc[df["category_name_1"] == 'Health & Sports']
KB = df.loc[df["category_name_1"] == 'Kids & Baby']


# In[130]:


de = Mobiles['Price Range'].value_counts()
dy = pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[131]:


x = Apl['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[132]:


x = SS['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[133]:


x = BG['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[134]:


x = Sogh['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[135]:


x = Others['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[136]:


x = HL['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[137]:


x = Ent['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[138]:


x = HS['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# In[139]:


x = KB['Price Range'].value_counts()
pd.DataFrame({'Price Range':x.index, 'Count':x.values})


# ## Month on Month Sales

# In[140]:


y = [2016.0]
m1 = ["July", "August", "September", "October", "November", "December"]
arr = []

for year in y:
    
    sub = df[df["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)


# In[141]:


y = [2017.0]
m1 = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

for year in y:
    
    sub = df[df["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)


# In[142]:


y = [2018.0]
m1 = ["January", "February", "March", "April", "May", "June", "July", "August"]

for year in y:
    
    sub = df[df["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)
        
print(arr)


# In[143]:


months = ["July 2016", "Aug 2016", "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016", "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "June 2017", "July 2017", "Aug 2017", "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017", "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "June 2018", "July 2018", "Aug 2018"]
sss = pd.DataFrame({'Months':months, 'Sales':arr})
sss


# In[144]:


fig = px.line(sss, x="Months", y="Sales", title='Month on Month Sales')
fig.update_traces(line_color='green')
fig.update_traces(mode='lines+markers')
fig.show()


# ## Order Prediction

# In[16]:


df


# In[17]:


df.columns


# In[18]:


ml = df[["status", "category_name_1", "payment_method", "qty_ordered", "BI Status", "Year", "Month Name", "Price Range", "discount_amount Range", "Customer ID"]]
ml


# In[19]:


columns_to_encode = ["status", "category_name_1", "payment_method", "BI Status", "Year", "Month Name", "Price Range", "discount_amount Range"]

# Initialize the label encoder
le = LabelEncoder()

# Loop through the columns and encode each one
for col in columns_to_encode:
    ml[col] = le.fit_transform(ml[col])

# Display the encoded DataFrame
ml


# In[20]:


from sklearn.decomposition import PCA

X = ml.values 
pca = PCA(n_components=10) 
X_pca = pca.fit_transform(X) 
X_pca


# In[21]:


ml["status"].value_counts()


# ### Extremely imbalanced data for this problem^^^

# In[22]:


temp = ml.drop(["status"], axis=1)
temp


# In[94]:


from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

x = temp
y = ml["status"]

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 123)

clf = LazyClassifier(verbose = 0, ignore_warnings = True, custom_metric = None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)


# ## Decision Tree Classifier

# In[ ]:


from sklearn.metrics import balanced_accuracy_score


# In[30]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = temp
y = ml["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

desc = "Decision Tree Classifier for Order Status Prediction"
mlflow.set_experiment("Order Status")
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.autolog(log_input_examples=True, log_model_signatures=True)

with mlflow.start_run(run_name="Decision Tree Classifier", description=desc) as run:
    dtc = DecisionTreeClassifier(random_state=42)

    dtc.fit(X_train, y_train)

    y_pred = dtc.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
#     balanced_acc = balanced_accuracy_score(y_test, predictions)
    
#     mlflow.log_metric("Balanced_Accuracy", balanced_acc)

    print("Accuracy: {:.2f}%".format(accuracy*100))


# ## Random Forest Classifier 

# In[28]:


from sklearn.ensemble import RandomForestClassifier

X = ml.drop('status', axis=1)
y = ml["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

desc = "Random Forest Classifier for Order Status Prediction"
mlflow.set_experiment("Order Status")
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.autolog(log_input_examples=True, log_model_signatures=True)

with mlflow.start_run(run_name="Random Forest Classifier", description=desc) as run:

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: {:.2f}%".format(accuracy*100))


# # Demand Forecasting

# ## Demand Forecasting for Mobiles & Tablets

# In[152]:


t2 = df[df['category_name_1'] == "Mobiles & Tablets"]


# In[153]:


y = [2016.0]
m1 = ["July", "August", "September", "October", "November", "December"]
arr = []

for year in y:
    
    sub = t2[t2["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)


# In[154]:


y = [2017.0]
m1 = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

for year in y:
    
    sub = t2[t2["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)


# In[155]:


y = [2018.0]
m1 = ["January", "February", "March", "April", "May", "June", "July", "August"]

for year in y:
    
    sub = t2[t2["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)
        
print(arr)


# In[156]:


months = ["July 2016", "Aug 2016", "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016", "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "June 2017", "July 2017", "Aug 2017", "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017", "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "June 2018", "July 2018", "Aug 2018"]
mon = [7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8]
ses = pd.DataFrame({'Months':mon, 'Sales':arr})
sss = pd.DataFrame({'Months':months, 'Sales':arr})
sss


# In[157]:


fig = px.line(sss, x="Months", y="Sales", title='Mobile Phones Sales History')
fig.show()


# ## SARIMAX

# In[158]:


import statsmodels.api as sm

train_endog = ses['Sales'].iloc[:13] 
train_exog = ses['Months'].iloc[:13]  

test_endog = ses['Sales'].iloc[-12:]  
test_exog = ses['Months'].iloc[-12:]  

model = sm.tsa.statespace.SARIMAX(endog=train_endog, exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))
fit_model = model.fit()

future_exog = test_exog  
future_preds = fit_model.forecast(steps=12, exog=future_exog)

rmse = ((test_endog - future_preds)**2).mean() ** 0.5
print(f'RMSE: {rmse:.2f}')

pred_df = pd.DataFrame({'Actual': test_endog, 'Predicted': future_preds}, index=test_endog.index)
fig = px.line(pred_df, x=pred_df.index, y=pred_df.columns, title='Sales Monthly Forecast')
fig.update_traces(mode='lines+markers')
fig.show()


# ## ARIMA

# In[159]:


cr = pd.DataFrame({'Sales':sss["Sales"]})

from statsmodels.tsa.arima.model import ARIMA

train = cr.iloc[:-12]
test = cr.iloc[-12:]

model = ARIMA(train, order=(1, 1, 1))
fit_model = model.fit()

preds = fit_model.predict(start=test.index[0], end=test.index[-1], type='levels')

rmse = ((test['Sales'] - preds) ** 2).mean() ** 0.5
print(f'RMSE: {rmse:.2f}')


# ## Random Forest Regressor

# In[160]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go

train_data = ses.sample(frac=0.6, random_state=1)
test_data = ses.drop(train_data.index)

X_train = train_data[['Months', 'Sales']].values
y_train = train_data['Sales'].values
X_test = test_data[['Months', 'Sales']].values
y_test = test_data['Sales'].values

model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse:.2f}')

trrr = last_eight = arr[-10:]

array1 = trrr
array2 = y_pred

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, 11)), y=array1, mode='lines', name='Array 1'))
fig.add_trace(go.Scatter(x=list(range(1, 11)), y=array2, mode='lines', name='Array 2'))
fig.update_layout(title='Sales Forecast', xaxis_title='Months', yaxis_title='Sales')
fig.update_traces(mode='lines+markers')
fig.show()


# ## Decision Tree Regressor

# In[161]:


from sklearn.tree import DecisionTreeRegressor

data = ses

train_data = data[:-12]
test_data = data[-12:]

model = DecisionTreeRegressor(max_depth=3)
model.fit(train_data[['Months']], train_data['Sales'])

predictions = model.predict(test_data[['Months']])

rmse = ((predictions - test_data['Sales'])**2).mean() ** 0.5
print(f'RMSE: {rmse:.2f}')

# plot the actual and predicted sales
pred_df = pd.DataFrame({'Actual': test_data['Sales'], 'Predicted': predictions}, index=test_data.index)
fig = px.line(pred_df, x=pred_df.index, y=pred_df.columns, title='Sales Monthly Forecast')
fig.update_traces(mode='lines+markers')
fig.show()


# In[162]:


print("RMSE per Regression Model for Mobiles & Tablets Category Forecasting: ")
print("SARIMAX - 3084.86")
print("ARIMA - 4831.54")
print("Random Forest Regressor - 2943.20")
print("Decision Tree Regressor - 2584.86")


# ## Demand Forecasting for Men's Fashion Category

# In[163]:


t2 = df[df['category_name_1'] == "Men's Fashion"]


# In[164]:


y = [2016.0]
m1 = ["July", "August", "September", "October", "November", "December"]
arr = []

for year in y:
    
    sub = t2[t2["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)


# In[165]:


y = [2017.0]
m1 = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

for year in y:
    
    sub = t2[t2["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)


# In[166]:


y = [2018.0]
m1 = ["January", "February", "March", "April", "May", "June", "July", "August"]

for year in y:
    
    sub = t2[t2["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)
        
print(arr)


# In[167]:


months = ["July 2016", "Aug 2016", "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016", "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "June 2017", "July 2017", "Aug 2017", "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017", "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "June 2018", "July 2018", "Aug 2018"]
mon = [7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8]
ses = pd.DataFrame({'Months':mon, 'Sales':arr})
sss = pd.DataFrame({'Months':months, 'Sales':arr})
sss


# In[168]:


fig = px.line(sss, x="Months", y="Sales", title="Men's Fashion Sales History")
fig.show()


# ## SARIMAX

# In[169]:


import statsmodels.api as sm

train_endog = ses['Sales'].iloc[:13] 
train_exog = ses['Months'].iloc[:13]  

test_endog = ses['Sales'].iloc[-12:]  
test_exog = ses['Months'].iloc[-12:]  

model = sm.tsa.statespace.SARIMAX(endog=train_endog, exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))
fit_model = model.fit()

future_exog = test_exog  
future_preds = fit_model.forecast(steps=12, exog=future_exog)

rmse = ((test_endog - future_preds)**2).mean() ** 0.5
print(f'RMSE: {rmse:.2f}')

pred_df = pd.DataFrame({'Actual': test_endog, 'Predicted': future_preds}, index=test_endog.index)
fig = px.line(pred_df, x=pred_df.index, y=pred_df.columns, title='Sales Monthly Forecast')
fig.update_traces(mode='lines+markers')
fig.show()


# ## ARIMA

# In[170]:


cr = pd.DataFrame({'Sales':sss["Sales"]})

from statsmodels.tsa.arima.model import ARIMA

train = cr.iloc[:-12]
test = cr.iloc[-12:]

model = ARIMA(train, order=(1, 1, 1))
fit_model = model.fit()

preds = fit_model.predict(start=test.index[0], end=test.index[-1], type='levels')

rmse = ((test['Sales'] - preds) ** 2).mean() ** 0.5
print(f'RMSE: {rmse:.2f}')


# ## Random Forest Regressor

# In[171]:


from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go

train_data = ses.sample(frac=0.6, random_state=1)
test_data = ses.drop(train_data.index)

X_train = train_data[['Months', 'Sales']].values
y_train = train_data['Sales'].values
X_test = test_data[['Months', 'Sales']].values
y_test = test_data['Sales'].values

model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse:.2f}')

trrr = last_eight = arr[-10:]

array1 = trrr
array2 = y_pred

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, 11)), y=array1, mode='lines', name='Array 1'))
fig.add_trace(go.Scatter(x=list(range(1, 11)), y=array2, mode='lines', name='Array 2'))
fig.update_layout(title='Sales Forecast', xaxis_title='Months', yaxis_title='Sales')
fig.update_traces(mode='lines+markers')
fig.show()


# ## Decision Tree Regressor

# In[172]:


from sklearn.tree import DecisionTreeRegressor

data = ses

train_data = data[:-12]
test_data = data[-12:]

model = DecisionTreeRegressor(max_depth=3)
model.fit(train_data[['Months']], train_data['Sales'])

predictions = model.predict(test_data[['Months']])

rmse = ((predictions - test_data['Sales'])**2).mean() ** 0.5
print(f'RMSE: {rmse:.2f}')

pred_df = pd.DataFrame({'Actual': test_data['Sales'], 'Predicted': predictions}, index=test_data.index)
fig = px.line(pred_df, x=pred_df.index, y=pred_df.columns, title='Sales Monthly Forecast')
fig.update_traces(mode='lines+markers')
fig.show()


# In[173]:


print("RMSE per Regression Model for Men's Fashion Category Forecasting: ")
print("SARIMAX - 2402.48")
print("ARIMA - 3971.09")
print("Random Forest Regressor - 1248.07")
print("Decision Tree Regressor - 1533.90")


# ## Demand Forecasting for Women's Fashion Category 

# In[174]:


t2 = df[df['category_name_1'] == "Women's Fashion"]


# In[175]:


y = [2016.0]
m1 = ["July", "August", "September", "October", "November", "December"]
arr = []

for year in y:
    
    sub = t2[t2["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)


# In[176]:


y = [2017.0]
m1 = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

for year in y:
    
    sub = t2[t2["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)


# In[177]:


y = [2018.0]
m1 = ["January", "February", "March", "April", "May", "June", "July", "August"]

for year in y:
    
    sub = t2[t2["Year"] == year]
    
    for month in m1:
        
        sub2 = sub[sub["Month Name"] == month]
        
        count = sub2["status"].shape[0]
        
        arr.append(count)
        
print(arr)


# In[178]:


months = ["July 2016", "Aug 2016", "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016", "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "June 2017", "July 2017", "Aug 2017", "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017", "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "June 2018", "July 2018", "Aug 2018"]
mon = [7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8]
ses = pd.DataFrame({'Months':mon, 'Sales':arr})
sss = pd.DataFrame({'Months':months, 'Sales':arr})
sss


# In[179]:


fig = px.line(sss, x="Months", y="Sales", title="Women's Fashion Sales History")
fig.show()


# ## SARIMAX

# In[180]:


import statsmodels.api as sm

train_endog = ses['Sales'].iloc[:13] 
train_exog = ses['Months'].iloc[:13]  

test_endog = ses['Sales'].iloc[-12:]  
test_exog = ses['Months'].iloc[-12:]  

model = sm.tsa.statespace.SARIMAX(endog=train_endog, exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))
fit_model = model.fit()

future_exog = test_exog  
future_preds = fit_model.forecast(steps=12, exog=future_exog)

rmse = ((test_endog - future_preds)**2).mean() ** 0.5
print(f'RMSE: {rmse:.2f}')

pred_df = pd.DataFrame({'Actual': test_endog, 'Predicted': future_preds}, index=test_endog.index)
fig = px.line(pred_df, x=pred_df.index, y=pred_df.columns, title='Sales Monthly Forecast')
fig.update_traces(mode='lines+markers')
fig.show()


# ## ARIMA

# In[181]:


cr = pd.DataFrame({'Sales':sss["Sales"]})

from statsmodels.tsa.arima.model import ARIMA

train = cr.iloc[:-12]
test = cr.iloc[-12:]

model = ARIMA(train, order=(1, 1, 1))
fit_model = model.fit()

preds = fit_model.predict(start=test.index[0], end=test.index[-1], type='levels')

rmse = ((test['Sales'] - preds) ** 2).mean() ** 0.5
print(f'RMSE: {rmse:.2f}')


# ## Random Forest Regressor

# In[182]:


from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go

train_data = ses.sample(frac=0.6, random_state=1)
test_data = ses.drop(train_data.index)

X_train = train_data[['Months', 'Sales']].values
y_train = train_data['Sales'].values
X_test = test_data[['Months', 'Sales']].values
y_test = test_data['Sales'].values

model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse:.2f}')

trrr = last_eight = arr[-10:]

array1 = trrr
array2 = y_pred

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, 11)), y=array1, mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=list(range(1, 11)), y=array2, mode='lines', name='Predicted'))
fig.update_layout(title='Sales Forecast', xaxis_title='Months', yaxis_title='Sales')
fig.update_traces(mode='lines+markers')
fig.show()


# ## Decision Tree Regressor

# In[183]:


from sklearn.tree import DecisionTreeRegressor

data = ses

train_data = data[:-12]
test_data = data[-12:]

model = DecisionTreeRegressor(max_depth=3)
model.fit(train_data[['Months']], train_data['Sales'])

predictions = model.predict(test_data[['Months']])

rmse = ((predictions - test_data['Sales'])**2).mean() ** 0.5
print(f'RMSE: {rmse:.2f}')


pred_df = pd.DataFrame({'Actual': test_data['Sales'], 'Predicted': predictions}, index=test_data.index)
fig = px.line(pred_df, x=pred_df.index, y=pred_df.columns, title='Sales Monthly Forecast')
fig.update_traces(mode='lines+markers')
fig.show()


# In[184]:


print("RMSE per Regression Model for Women's Fashion Category Forecasting: ")
print("SARIMAX - 1765.43")
print("ARIMA - 1708.88")
print("Random Forest Regressor - 757.16")
print("Decision Tree Regressor - 1147.17")


# # Fin - Till FYP1 Mid1
