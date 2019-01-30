Guidelines to run the program:

The code is developed using python3 libraries and will require pandas, numpy, statsmodel, and matlibplot to successfully run the program and visualize the results. The following commands should be used to run the program.

$ python submission.py

It will write the prediction.txt and store the png images of each dataset.


Introduction:

Time series prediction is an important aspect of Data Mining, as it can provide valuable information about the future sales and customer behavior. The companies can leverage this tool to strategically allocate their resources and identify potential customers. 

Objective:

In this project, we have been given sales data of 100 key products over 118 days. It also includes the details of customers including their ages. We have been tasked to forecast the individual sales of 100 key products and combined sales of 100 products for next 29 days. 

Method:

I have developed a python based model to forecast the time series in order to complete this project. The process of model development is divided into three parts:

• Identification: Since, the sales data demonstrate a combination of oscillations and upward and downward trends, a robust model that can handle seasonality, non-stationary data is required for forecasting. I have gone through various resources including Wikipedia, reference book, and course material to identify SARIMAX as a suitable model for this project. I have also investigated ARIMA model, however, it was challenging to converge and obtain reasonable forecasting results.

• Training and testing: The training is the part where I identify the suitable parameters for the SARIMAX model. Initially, I have divided the data into two sets: training and testing. The first 90% of sales data is used for training and rest are used to testing of the model. I have employed a grid search approach to identify the suitable parameters for the model. In this grid search approach, I pick a combination of (p,d,q) and build the respective model to fit the training data. Then the model is used to forecast for the testing data and an error residual is calculated between forecasted and testing data. A looping is performed over (p,d,q), and the (p,d,q) associated with the minimum residual between forecasted and actual testing data is picked as the optimum model parameter for the give dataset. 

• Forecasting: The model parameter obtained from previous step is used for forecasting 29 days in future. The approach is iteratively applied to all 101 data sets to obtain their forecast. 

Implementation:

The implementation is mainly divided in six processes

1) Loading of data file: The two data files "key_product_IDs.txt" and "product_distribution_training_set.txt" are loaded into code. 
2) Extraction of training and testing data sets: Data is divided in two data sets: training (90%) and testing (10%).
3) Stationary check: Rolling mean and standard deviation plots are utilized to visualize the stationary nature of data.
4) Seasonality check: seasonal_decompose is performed to visualize the seasonality of data.
5) Grid search approach to obtain (p,d,q): As described earlier, we find the optimum parameters for the SARIMAX model using the grid search approach. Iteratively, p is varied from 1 to 4, d is varied from 0 to 1, and q is varied from 1 to 4.
6) Forecasting for 29 days: (p,q,d) obtained from grid search is used to forecast for next 29 days.
7) Writing of forecast data: Data is rearranged to write in required format as described in the project description.

If you have any question feel free to reach me at aaruj1@binghamton.edu

