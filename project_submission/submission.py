"""
Guidelines to run the program:
The code is developed using python3 libraries and will require pandas, numpy, 
statsmodel, and matlibplot to successfully run the program and visualize the 
results. The following commands should be used to run the program.

   $ python submission.py

It will write the prediction.txt and store the png images of each dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from statsmodels.tsa.statespace.sarimax import SARIMAX
style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")

"""
################################################################
################### FUNCTION CALLS #############################
################################################################
"""
#Stationary check: Rolling mean and standard deviation plots are utilized to 
#visualize the stationary nature of data.
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=7).mean()
    rolstd = timeseries.rolling(window=7).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(8,6), dpi=150)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = [11, 4]
    plt.plot(timeseries, '.-', color='blue', label='Original', linewidth=1.0)
    plt.plot(rolmean, '.-', color='black', label='Rolling Mean', linewidth=1.0)
    plt.plot(rolstd, '.-', color='red', label = 'Rolling Std', linewidth=1.0)
#    plt.xticks(rotation='vertical')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    filename = 'stationarity.png'
    fig.savefig(filename, bbox_inches='tight')
    plt.show(block=False)
    
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#Seasonality check: seasonal_decompose is performed to visualize the 
#seasonality of data.

def check_seasonality(data):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(data)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    data_plot = data.copy();
    data_plot.index = np.arange(1,len(data_plot)+1)
    
    trend_plot = trend.copy();
    trend_plot.index = np.arange(1,len(trend_plot)+1)
    
    seasonal_plot = seasonal.copy();
    seasonal_plot.index = np.arange(1,len(seasonal_plot)+1)
    
    residual_plot = residual.copy();
    residual_plot.index = np.arange(1,len(residual_plot)+1)
    
    
    
    fig = plt.figure(figsize=(8,6), dpi=150)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = [11, 22]
    plt.subplot(411)
    plt.plot(data_plot, '.-', color='blue', label='Original', linewidth=1.0)
#    plt.xticks(rotation='vertical')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend_plot, '.-', color='black', label='Trend', linewidth=1.0)
#    plt.xticks(rotation='vertical')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal_plot, '.-', color='red',label='Seasonality', linewidth=1.0)
#    plt.xticks(rotation='vertical')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual_plot, '.-', color='green', label='Residuals', linewidth=1.0)
#    plt.xticks(rotation='vertical')
    plt.legend(loc='best')
    plt.tight_layout()
    filename = 'seasonality.png'
    fig.savefig(filename, bbox_inches='tight')
    plt.show(block=False)


#Grid search approach to obtain (p,d,q): As described earlier, we find the 
#optimum parameters for the SARIMAX model using the grid search approach. 
#Iteratively, p is varied from 1 to 4, d is varied from 0 to 1, and q is varied
#from 1 to 4.
def findModelParameters(trainData,testData):
    min = 100000
    
    for i in range(1,5): # looping over p
        for j in range(1,5): # looping over q
            for k in range (0,2): # looping over d
                if i == 0 and j == 0:
                    continue
                else :
                    stepwise_model = SARIMAX(trainData, order=(i,k,j), seasonal_order=(2,0,0,7), enforce_stationarity=False, enforce_invertibility=False)
                    try:
                        results_SARIMAX = stepwise_model.fit(disp=-1)
                    except:
                        continue
#                    future_forecast = results_SARIMAX.forecast(len(testData)).astype(int)
#                    RSS = (sum((future_forecast-testData)**2))
                    RSS = (sum((results_SARIMAX.fittedvalues-trainData)**2)**0.5)
                    print('ARIMA(%d,0,%d) ==> AIC :: %2f, BIC :: %2f, RSS :: %2f' % (i,j,results_SARIMAX.aic, results_SARIMAX.bic, RSS))
                    if (RSS < min):
                        min = RSS
                        p = i
                        q = j
                        d = k
    print('\nselected model :: SARIMAX(%d,%d,%d) \n' % (p,d,q))
    return (p,d,q)

# Import key product IDs and product distribution training set
key_product_IDs = pd.read_table("./inputdata/key_product_IDs.txt", header=None)
product_distribution_training_set = pd.read_table("./inputdata/product_distribution_training_set.txt", header=None)
product_distribution_training_set = product_distribution_training_set.T

product_distribution_training_set = product_distribution_training_set.drop(product_distribution_training_set.index[0])
product_distribution_training_set.index = pd.date_range(start='11/1/2018', periods=118)
product_distribution_training_set.index = pd.to_datetime(product_distribution_training_set.index)

# Make entry for total sales numbers
key_product_IDs.columns = ["key_product_ID"]

key_product_IDs = key_product_IDs.set_value(len(key_product_IDs), 'key_product_ID', 0)

# Calculation the total sale for each day and store in the last column of dataframe
product_distribution_training_set['0'] = product_distribution_training_set[product_distribution_training_set.columns].sum(axis=1)

writeflag = 0

"""
##############################################################
################### MAIN PROGRAM #############################
##############################################################
"""
# Iterate over all products and forecast for 29 days
for iproduct in range(100,101):
    
    iproduct = 100;
    
    
    # Extract the training and testing data for the current product
    print('forecasting for product ID %d' %(key_product_IDs.iloc[iproduct,0].astype(int)))
    
    trainStartIndex = 0
    trainEndIndex = np.floor(len(product_distribution_training_set.index)*0.9).astype(int)
    testStartIndex = trainEndIndex
    testEndIndex = len(product_distribution_training_set.index)
    
    sampleData= product_distribution_training_set.iloc[trainStartIndex:testEndIndex, iproduct]
    trainData = product_distribution_training_set.iloc[trainStartIndex:trainEndIndex, iproduct]
    testData = product_distribution_training_set.iloc[testStartIndex:testEndIndex, iproduct]
    
    sampleData_plot = sampleData.copy();
    sampleData_plot.index = np.arange(1,len(sampleData_plot)+1)

#    sampleData_plot.reset_index(drop=True)
    trainData_plot = trainData.copy();
    trainData_plot.index = np.arange(1,len(trainData_plot)+1)
    testData_plot = testData.copy();
    testData_plot.index = np.arange(len(trainData_plot)+1,len(trainData_plot)+len(testData_plot)+1)
    
    #plot the original, fitted and forecast data
    fig = plt.figure(figsize=(8,6), dpi=150)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = [11, 4]
    plt.plot(sampleData, '.-', color='blue', label='Original data', linewidth=1.0)
    plt.legend(loc='best')
    title = 'Original data for overall product'
    plt.title(title)
#    plt.xticks(rotation='90')
    filename = 'overall_all_original.png'
    fig.savefig(filename, bbox_inches='tight')

    #plot the original, fitted and forecast data
    fig = plt.figure(figsize=(8,6), dpi=150)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = [11, 4]
#    plt.plot(sampleData, '.-', color='blue', label='Original data', linewidth=1.0)
    plt.plot(trainData_plot, '.-', color='black', label='training data', linewidth=1.0)
    plt.plot(testData_plot, '.-', color='red', label='testing data', linewidth=1.0)
    plt.legend(loc='best')
    title = 'Original data for overall product'
    plt.title(title)
#    plt.xticks(rotation='90')
    filename = 'overall_all_traintest.png'
    fig.savefig(filename, bbox_inches='tight')
    
    test_stationarity(sampleData_plot)
    check_seasonality(sampleData)

    
    
    #find most suitable model for the prediction using 
    print('Performing the grid search to obtain (p,d,q)')
    p,d,q = findModelParameters(sampleData,testData)
    
    # use the obtained (p,d,q) to fit the total data
    stepwise_model = SARIMAX(sampleData, order=(p,d,q), seasonal_order=(2,0,0,7), enforce_stationarity=False, enforce_invertibility=False)
    results_SARIMAX = stepwise_model.fit(disp=-1)
    
    # forecast for next 29 days
    future_forecast = results_SARIMAX.forecast(29).astype(int)
    future_forecast = future_forecast.astype(int)
    future_forecast[future_forecast < 0] = 0
    
    # Print the forecast data with day
    day = 118
    
    for yhat in future_forecast:
    	print('Day %d: %d' % (day, yhat))
    	day += 1
    print('\n\n')
#    column_name = 'key_product_' + str(key_product_IDs.iloc[iproduct,0].astype(int))
    future_forecast = pd.DataFrame(future_forecast,columns=[key_product_IDs.iloc[iproduct,0].astype(int)])
    future_forecast.index = pd.date_range(start='2/26/2019', periods=29)
    

#     plot the original, fitted and forecast data
    future_forecast_plot = future_forecast.copy()
    future_forecast_plot.index = np.arange(len(sampleData)+1,len(sampleData)+len(future_forecast_plot)+1)
    results_SARIMAX_fitted_data_plot = results_SARIMAX.fittedvalues.copy()
    results_SARIMAX_fitted_data_plot.index = np.arange(1,len(results_SARIMAX_fitted_data_plot)+1)
    fig = plt.figure(figsize=(8,6), dpi=150)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = [11, 4]
    plt.plot(sampleData_plot, '.-', color='blue', label='Original data', linewidth=1.0)
    plt.plot(results_SARIMAX_fitted_data_plot, '.-', color='red', label='predicted', linewidth=1.0)
    plt.plot(future_forecast_plot, '.-', color='black', label='29 days forecast', linewidth=1.0)
    plt.legend(loc='best')
    title = 'Original data and 29 days forecast for product ' + str(key_product_IDs.iloc[iproduct,0].astype(int))
    plt.title(title)
    plt.xticks(rotation='90')
    filename = 'product_' + str(key_product_IDs.iloc[iproduct,0].astype(int)) + '_forecast.png'
    fig.savefig(filename, bbox_inches='tight')
    
    #concatnate all the prediction into one dataframe
    if writeflag == 0 :
#        days = pd.DataFrame(index=future_forecast.index)
#        days['day'] = 118
#        for i in range(29):
#            days.iloc[i] = 118+i
#        
#        
#        predicted = days
        predicted_combined = future_forecast
        writeflag = 1
    else:
        predicted_combined = pd.concat([predicted_combined, future_forecast], axis=1)


#reformat the data for writing to a text file in desired format
sum_col = predicted_combined[0]
predicted_combined.drop(labels=[0], axis=1,inplace = True)
predicted_combined.insert(0, 0, sum_col)

predicted_combined = predicted_combined.T
predicted_combined.to_csv('prediction.txt', sep=' ', encoding='utf-8', header=False)

################################################################
################### END OF PROGRAM #############################
################################################################
