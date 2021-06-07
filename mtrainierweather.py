import pandas as pd 
import numpy as np
import datetime as dt
#import sweetviz as sv
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
#import seaborn as sns

rw = pd.read_csv('Rainier_Weather.csv', index_col=0, low_memory=False, parse_dates=True)
rw = rw.reindex(index=rw.index[::-1])
rw = rw.reset_index()
rwshape = rw.shape
rw = rw.round(2)
#R: 464, C: 7

rw['Date'] = rw['Date'].dt.strftime('%d-%m-%Y')
date = rw['Date']
#Cleaned, This is data from 09-23-2014 to 12-31-2015
voltavg = rw['Battery Voltage AVG']
#in volts
tempavg = rw['Temperature AVG']
#in Fahrenheit
relhumavg = rw['Relative Humidity AVG']
#in %
windspdavg = rw['Wind Speed Daily AVG']
#assuming in mph
winddiravg = rw['Wind Direction AVG']
#in  circular degrees
solradavg = rw['Solare Radiation AVG']
#unsure about the unit measurements here, so we will not address this.

#Undo the statements below for the report, and comment out the date stuff!
"""rw_report = sv.analyze(rw)
rw_report.show_html('Rainier_Weather.html')"""

[avgtemp, medtemp, hitemp, lotemp] = [tempavg.mean(), tempavg.median(), tempavg.max(), tempavg.min()]
tempstuff = [avgtemp, medtemp, hitemp, lotemp]
#Average temperature, 28.02 deg F
#Median temperature, 27.36 deg F
#Highest temperature, 56.15 deg F
#Lowest temperature, -0.17 deg F
#The most common temp, 25.92 deg F
[avgwinddir, medwinddir, mowinddir] = [winddiravg.mean(), winddiravg.median(), winddiravg.mode()]
winddirstuff = [avgwinddir, medwinddir, mowinddir]
#Average wind direction, 199.44 deg(or SSE)
#Median wind direction, 227.47 deg(or ESE)
#Mode wind direction, 224.60 deg(or SE)
[avgspeed, medspeed, hispeed, lospeed, mospeed] = [windspdavg.mean(), windspdavg.median(), windspdavg.max(), windspdavg.min(), windspdavg.mode()]
windspdstuff = [avgspeed, medspeed, hispeed, lospeed, mospeed]
#AVG: 13.02 mph
#MED: 8.71 mph
#HI: 65.85 mph
#LO: 0.0 mph
#MODE: 0.0mph
[avghum, medhum, hihum, lohum, mohum] = [relhumavg.mean(), relhumavg.median(), relhumavg.max(), relhumavg.min(), relhumavg.mode()]
relhumstuff = [avghum, medhum, hihum, lohum, mohum]
#AVG: 62.5 %
#MED: 64.1 %
#HI: 100.0 %
#LO: 9.7 %
#MODE: 100.0 %

fixed_date = date[100:464]
fixed_date = fixed_date.reset_index(drop=True)
fixed_temp = tempavg[100:464]
fixed_temp = fixed_temp.reset_index(drop=True)
rwindex = pd.Series(rw.index[0:364])
rwindex16 = rwindex + 364
#datefuture = date + pd.offsets.DateOffset(days=365)
#I decided this is important enough to save here. I will gather other stuff here, too.

#Q1: What will the temperature look like for 01-01-2016 to 12-31-2016?
#This is an experiment with linear regression

[m, bl] = np.polyfit(rwindex, fixed_temp, 1)
modtempd1 = [m, bl]
#print(modtempd1)
y1 = (m * rwindex) + bl
y1 = y1.round(2)
y1dif = abs(fixed_temp - y1)
y1err = y1dif / fixed_temp
#Yay, my first step towards regression! How about a degree 2 poly?
#Also made % error using y1dif and y1err

[a2, b2, c2] = np.polyfit(rwindex, fixed_temp, 2)
modtempd2 = [a2, b2, c2]
#print(modtempd2)
y2 = (a2 * (rwindex ** 2)) + (b2 * rwindex) + c2
y2 = y2.round(2)
y2dif = abs(fixed_temp - y2)
y2err = y2dif / fixed_temp
#Parabolic Regression

[a3, b3, c3, d3] = np.polyfit(rwindex, fixed_temp, 3)
modtempd3 = [a3, b3, c3, d3]
#print(modtempd3)
y3 = (a3 * (rwindex ** 3)) + (b3 * (rwindex ** 2)) + (c3 * rwindex) + d3
y3 = y3.round(2)
y3dif = abs(fixed_temp - y3)
y3err = y3dif / fixed_temp
#Cubic Regression

#Now I want to start predicting for the year 2016:
x_1= np.array(rwindex16).reshape(-1, 1)
y_1 = np.array(fixed_temp)
#model = LinearRegression().fit(x_1, y_1)
#r_sq = model.score(x_1, y_1)
#print('Coefficient of determination:', r_sq)
#y1_pred = model.predict(x_1)
#print('Predicted Response: ', y1_pred, sep='\n')
#Linear Regression Prediction Model, with a mini machine

#Degree 3 Polynomial Regression Prediction Model?
#Look at the xtrans line to figure out what's throwing an error there
"""x_3 = np.array(rwindex16).reshape(-1, 1)
y_3 = np.array(fixed_temp)
xtrans = PolynomialFeatures(degree=3, include_bias=False).fit_transform(x_3)
model3 = LinearRegression.fit(xtrans, y_3)
r_sq3 = model3.score(xtrans, y_3)
y3_pred = model3.predict(xtrans)"""

my_ticks = ['01-01-2015',
            '01-02-2015',
            '01-03-2015',
            '01-04-2015',
            '01-05-2015',
            '01-06-2015',
            '01-07-2015',
            '01-08-2015',
            '01-09-2015',
            '01-10-2015',
            '01-11-2015',
            '01-12-2015']
my_ticks = pd.Series(my_ticks)
my_ticks = pd.to_datetime(my_ticks, format='%d-%m-%Y').dt.strftime('%d-%m-%Y')
labels = ['Actual AVG Temp', 'Linear Fit', 'Parabolic Fit', 'Cubic Fit']

my_ticks_2016 = ['01-01-2016',
            '01-02-2016',
            '01-03-2016',
            '01-04-2016',
            '01-05-2016',
            '01-06-2016',
            '01-07-2016',
            '01-08-2016',
            '01-09-2016',
            '01-10-2016',
            '01-11-2016',
            '01-12-2016']
my_ticks_2016 = pd.Series(my_ticks_2016)
my_ticks_2016 = pd.to_datetime(my_ticks_2016, format='%d-%m-%Y').dt.strftime('%d-%m-%Y')

fig1 = plt.figure(1)
plt.plot_date(fixed_date, fixed_temp)
plt.plot(rwindex, y1)
plt.plot(rwindex, y2)
plt.plot(rwindex, y3)
plt.title('Daily Avg Temp, Mt. Rainier, 2015')
plt.xlabel('Date')
plt.xticks(rotation=45, ticks=my_ticks)
plt.legend(labels)
plt.ylabel('Temp, in deg F')

fig2 = plt.figure(2)
plt.plot_date(fixed_date, y1err, 'orange')
plt.plot(rwindex, y2err, 'green')
plt.plot(rwindex, y3err, 'red')
plt.xticks(rotation=45, ticks=my_ticks)
plt.title('Daily Average Temperature, Percent Errors, Mt. Rainier, 2015')
plt.legend(labels[1:4])

"""fig3 = plt.figure(3)
plt.plot(x_1, y1_pred, 'orange')
#plt.plot(xtrans, y3_pred, 'red')
#plt.plot(rwindex, y3, 'red')
plt.title('Predictions for Mt. Rainier Weather Data, 2016')
plt.legend(labels[1:4])

plt.show()"""
