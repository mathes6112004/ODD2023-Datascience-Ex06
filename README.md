# ODD2023-Datascience-Ex06
## AIM
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM
### STEP 1:
Read the given Data

### STEP 2:
Clean the Data Set using Data Cleaning Process

### STEP 3:
Apply Feature Transformation techniques to all the features of the data set

### STEP 4:
Print the transformed features

## PROGRAM:
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
```
```
from google.colab import files
uploaded = files.upload()
```
```
df = pd.read_csv('Data_to_Transform(1).csv')
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/eb86b376-6b25-4c25-82e1-2bbb160318a5)
```
df.skew()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/663e09fc-81bf-4130-8699-b40b71c0b524)
```
df1 = df.copy()
sm.qqplot(df1['HighlyPositiveSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/58985153-606f-433f-b8bb-87150ed276bf)
```
sm.qqplot(df1['HighlyNegativeSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/ec401ace-5e74-4c40-a325-4b14c9848f3a)
```
sm.qqplot(df1['HighlyNegativeSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/2afc8b32-9bb3-4b1e-af67-01e87a96c9b8)
```
sm.qqplot(df1['ModerateNegativeSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/468ca9d2-b0e7-496b-9636-be4e3f3d9a8e)
```
df1['HighlyPositiveSkew'] = np.log(df1['HighlyPositiveSkew'])
sm.qqplot(df1['HighlyPositiveSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/fac3b6ab-eb78-4b0c-8520-05f1cdd6e742)
```
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2['HighlyPositiveSkew']
sm.qqplot(df2['HighlyPositiveSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/138f15dc-7636-4ddd-a9e4-e7d143338069)
```
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3['HighlyPositiveSkew']**(1/1.2)
sm.qqplot(df2['HighlyPositiveSkew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/d131af8a-6e67-46c6-99be-f65eaf4c39a1)
```
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4['ModeratePositiveSkew'])
sm.qqplot(df4['ModeratePositiveSkew_1'],fit=True,line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/40c6b578-7a14-4564-ab99-82dd545a8f7f)
```
from sklearn.preprocessing import PowerTransformer
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/b4d2c251-6f08-4095-b232-86ec60f2b3f9)
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex06/assets/119477782/1ee0f53f-c786-477d-b7bb-1ce9f09ee2fc)
# RESULT:
Thus feature transformation is done for the given set
