###import####
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from matplotlib import pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate

# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

warnings.simplefilter(action='ignore', category=Warning)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 100)


######################################################
#Görev1 : Kesifci Veri Analizi
######################################################
df=pd.read_excel("Res_Final_R1.xlsx")
df.head()
def check_df(dataframe, head=5):
    print("############### Shape ################")
    print(dataframe.shape)
    print("########### Types ###############")
    print(dataframe.dtypes)
    print("########### Head ###############")
    print (dataframe.head(head))
    print ("########### Tail ###############" )
    print ( dataframe.tail(head))
    print ( "########### NA ###############" )
    print ( dataframe.isnull().sum())
    print ( "########### Quantiles ###############" )
    print ( dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T )
    print ( "########### Nunique ###############" )
    print ( dataframe.nunique())

check_df(df)
df.info()
######Degisken Türlerini Belirleme#########


from datetime import datetime

df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y-%m-%d %H:%M:%S")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#######################################################
#Numerik ve kategorik değişkenlerin veri içindeki dağılımı
#######################################################

df.corr()


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

for col in num_cols:
    plot_numerical_col(df, col)

####################################################
#Kategorik değişkenler ile hedef değişken incelemesi
####################################################
def target_summary_with_cat(dataframe, target, cat_col):
    print(dataframe.groupby(cat_col).agg({target: "mean"}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Active_Power", col)

##en yüksek üretim ortalama da agustos ayi(2336) daha sonra haziran (1988)
##kis aylari neden yok dogalgaz calistigi icin bilincli mi calistirilmiyor????

############################################
#Aykırı gözlem analizi
############################################

def box_numerical_col(dataframe, numerical_col):
    sns.boxplot(x=df[numerical_col])
    plt.show(block=True)

box_numerical_col(df,"Active_Power")

for col in num_cols:
    box_numerical_col(df, col)

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

##Aykiri deger yok!!!!!!!!!!!!

######################################
#Eksik gözlem analizi
######################################
df["Active_Power"].replace(0, np.nan, inplace=True)
df.isnull().values.any()
#True
df=df.dropna()

###active power 0 olan ama diger degiskenlerin degerleri dolu olanlar var bunlari incelemek gerekebilir.Veri setinden cikarilabilir.


####################################################
#Base Model icin agac yöntemlerini deneyelim(daha sonra feature engineering yapip tekrar model degerlendirmesi yapacagiz)
####################################################
df=df.drop(['Date',"DateTime","Time"], axis = 1)
df.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

##standartlastirma yapacagimiz icin bagimli degiskenimizi numerik kolanlar listesinden cikarmaliyiz.
num_cols = [col for col in num_cols if col not in "Active_Power"]

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
#df1[num_cols] = rs.inverse_transform(df1[num_cols])
df.head()

df1[num_cols].head()
df1[num_cols].dtypes #float

scaler = StandardScaler()
df1[num_cols] = scaler.fit_transform(df1[num_cols])

####Base Model Kurma######

X = df.drop("Active_Power", axis = 1)
y = df[["Active_Power"]]

X_train=X.iloc[:16587,:]
X_test=X[16587:]
y_train=y.iloc[:16587,:]
y_test=y[16587:]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

models = [('LR', LinearRegression()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, modell in models:
    rmse = np.mean(np.sqrt(-cross_val_score(modell, X, y, cv=5, scoring="neg_mean_squared_error"))) ## 10 Katlı CV RMSE
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


gbm_model =GradientBoostingRegressor()

gbm_model.fit(X_train, y_train)

y_pred=gbm_model.predict(X_test)
y_test

gbm_model.get_params()

rmse = np.mean(np.sqrt(-cross_val_score(gbm_model,
                                        X_test, y_test, cv=5, scoring="neg_mean_squared_error")))

#rmse=23.74

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

y_pred= pd.DataFrame(y_pred)
y_predd=np.array(y_pred)
y_testt=np.array(y_test)

y_pred.to_excel("y_pred.xlsx",index=False)
y_test.to_excel("y_gercek.xlsx",index=False)
#SMAPE Hesabi

smape(y_predd,y_testt)
#49.98

######Base Model Önemi######

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(9, 9))
    sns.set(font_scale=0.4)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(gbm_model,X,num=10)
#wind speed en önemli degisken

#######################################
###Feature Engineering###
########################################

df=pd.read_excel("Res_Final_R1.xlsx")
df.head()
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y-%m-%d %H:%M:%S")

df["Active_Power"].replace(0, np.nan, inplace=True)
df=df.dropna()
df.info()

import math
df["New_Horizontal_WS"]=df["windspeed_100m"]*np.cos(df["winddirection_100m"]*3.14/180)
df["New_Vertical_WS"]=df["windspeed_100m"]*np.sin(df["winddirection_100m"]*3.14/180)

df["New_Sin_Time"]=np.sin(df["Time"])
df["New_Cos_Time"]=np.cos(df["Time"])

def create_date_features(df):
    df['month'] = df.Date.dt.month
    df['day_of_month'] = df.Date.dt.day
    df['day_of_year'] = df.Date.dt.dayofyear
    df['week_of_year'] = df.Date.dt.weekofyear
    df['day_of_week'] = df.Date.dt.dayofweek
    df['is_month_start'] = df.Date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.Date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
df.head()

df.loc[(df['windspeed_100m'] < 1), 'NEW_WS_CATEGORY'] = 'Calm'
df.loc[(df['windspeed_100m'] >1) & (df['windspeed_100m'] <= 5), 'NEW_WS_CATEGORY'] = 'LightAir'
df.loc[(df['windspeed_100m'] >5) & (df['windspeed_100m'] <= 11), 'NEW_WS_CATEGORY'] = 'LightBreeze'
df.loc[(df['windspeed_100m'] >11) & (df['windspeed_100m'] <= 19), 'NEW_WS_CATEGORY'] = 'GentleBreeze'
df.loc[(df['windspeed_100m'] >19) & (df['windspeed_100m'] <= 28), 'NEW_WS_CATEGORY'] = 'ModerateBreeze'

df.head()
df.info()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

##Model Kurma##


cat_cols, num_cols, cat_but_car = grab_col_names(df)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

##standartlastirma yapacagimiz icin bagimli degiskenimizi numerik kolanlar listesinden cikarmaliyiz.
num_cols = [col for col in num_cols if col not in ["Active_Power","DateTime","Date","Time"]]

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
#df1[num_cols] = rs.inverse_transform(df1[num_cols])
df.head()

###Feature Sonrasi yeni model####

X = df.drop(["Active_Power",'Date',"DateTime","Time"], axis = 1)
y = df[["Active_Power"]]
X_train=X.iloc[:16587,:]
X_test=X[16587:]
y_train=y.iloc[:16587,:]
y_test=y[16587:]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


gbm_model =GradientBoostingRegressor()

gbm_model.fit(X_train, y_train)

y_pred=gbm_model.predict(X_test)
y_test


rmse = np.mean(np.sqrt(-cross_val_score(gbm_model,
                                        X_test, y_test, cv=5, scoring="neg_mean_squared_error")))

#rmse=23.1531

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

y_pred= pd.DataFrame(y_pred)
y_predd=np.array(y_pred)
y_testt=np.array(y_test)

#y_pred.to_excel("y_pred.xlsx",index=False)
#y_test.to_excel("y_gercek.xlsx",index=False)
#SMAPE Hesabi

smape(y_predd,y_testt)
#50.1801

######Feature Agac Model Önemi######

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(9, 9))
    sns.set(font_scale=0.4)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(gbm_model,X,num=10)

































