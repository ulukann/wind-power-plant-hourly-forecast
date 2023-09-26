#############################################
#import islemleri
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate

# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib

import warnings
import joblib
import pydotplus
from skompiler import skompile
import graphviz

warnings.simplefilter(action='ignore', category=Warning)


# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 100)


import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
from datetime import date

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

#####################################################
# Active Power Forecasting
#####################################################
# #Veri Seti Hikayesi##
# 39.2841480 enleminde 27.6418380 boylaminda Ege bölgesinde bulunan Bilgin Enerji siketine ait
# 46 adet rüzgar tirübününün 01.01.2020-31.07.2023 dönemi arasindaki saatlik üretim bilgisini icermektedir.
#Veri seti icerisinde ilgili tarih ve saat bilgisine ait asagidaki degiskenler bulunmaktadir.
#Elektrik Piyasası’nın en temel amacı, depolanamayan elektrik enerjisinin mümkün olan en az kayıpla
# ihtiyaç duyulduğu kadar üretilip tüketilmesini sağlamaktır.Bu amaç doğrultusunda Elektrik Piyasası, Gün Öncesi Piyasası (GÖP),
# Gün İçi Piyasası (GİP) ve Dengeleme Güç Piyasası(DGP) ile piyasa faaliyetlerini gerçekleştirmektedir.
#Gün İçi Piyasası (GİP): GİP elektriğin teslimat saatinden 90 dakika öncesinde elektrik ticareti ve dengeleme faaliyetleri için kullanılan,
#Piyasa İşletmecisi (EPİAŞ) tarafından işletilen, organize edilen bir piyasadır.
#Biz de sirket icin önemli olan saatlik tahmin hesaplamasi icin en az hatayi veren modeli hesaplamak istiyoruz.

#Date:Gun-ay-yil tarih bilgisini icerir.
#Time:Saat bilgisi
#Active_Power:Ilgili tarih saat dilimi icerisinde sistemden cekilen üretilen enerji miktari
#Availability:Santrallerin ilgili saat diliminde max üretebilecegi üretim miktari
#temperature_2m (°C):Ilgili tarih saat dilimi icerisinde 2 m hava  deki hava sicakligi (°C)
#apparent_temperature (°C):Ilgili tarih saat dilimi icerisinde bölgedeki hava sicakligi (°C)
#rain (mm):
#surface_pressure (hPa):Ilgili tarih saat dilimi icerisinde bölgedeki hava basinci(hPa)
#windspeed_10m(km/h):Ilgili tarih saat dilimi icerisinde 10 m deki Rüzgarin hizi((km/h)
#windspeed_100m(km/h):Ilgili tarih saat dilimi icerisinde 100 m deki Rüzgarin hizi((km/h)
#winddirection_10m(°):Ilgili tarih saat dilimi icerisinde 10 m deki Rüzgarin yönü(°)
#winddirection_100m(°):Ilgili tarih saat dilimi icerisinde 100 m deki Rüzgarin yönü(°)
#soil_temperature_0_to_7cm(°C):Yer altındaki 0_to_7cm(°C) toprak seviyesinin ortalama sıcaklığı.
#soil_temperature_7_to_28cm(°C):Yer altındaki 7_to_28cm(°C) toprak seviyesinin ortalama sıcaklığı.
#soil_temperature_28_to_100cm(°C):Yer altındaki 28_to_100cm(°C) toprak seviyesinin ortalama sıcaklığı.
#soil_temperature_100_to_255cm(°C):Yer altındaki 100_to_255cm(°C) toprak seviyesinin ortalama sıcaklığı.
#soil_moisture_0_to_7cm(m³/m³):0-7 cm derinliklerde hacimsel karışım oranı olarak ortalama toprak su içeriği.
#soil_moisture_7_to_28cm(m³/m³):7-28 cm derinliklerde hacimsel karışım oranı olarak ortalama toprak su içeriği.
#soil_moisture_28_to_100cm(m³/m³):28-100 cm derinliklerde hacimsel karışım oranı olarak ortalama toprak su içeriği.
#soil_moisture_100_to_255cm(m³/m³):100-255 cm derinliklerde hacimsel karışım oranı olarak ortalama toprak su içeriği.

 #Toplam da 24 degiskenimiz bulunmaktadir.

########################################################
##Veri Setini Indirme
########################################################
df=pd.read_excel("Res_Final_R1.xlsx")
df.head()
df.info()

######################################################
#Kesifci Veri Analizi
######################################################
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

#Veri setimizde tarihler haric tüm degiskenlerimiz numeric tipte gelmektedir.
#Tarih degiskenlerimizin tiplerini datetime a cevirecegiz.

df.info()
df.shape
#shape: (22608, 24)

from datetime import datetime

df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y-%m-%d %H:%M:%S")

###########################################
######Degisken Türlerini Belirleme#########

def grab_col_names(dataframe, cat_th=12, car_th=20):
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
#Eksik Gözlem Analizi ve Degiskenlerin veri içindeki dağılımı
#######################################################
df["Active_Power"].replace(0, np.nan, inplace=True)
df.isnull().values.any()
#True
df=df.dropna()
df.isnull().values.any()
#False
df.shape
#(20734, 20)


def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

for col in num_cols:
    plot_numerical_col(df, col)

def box_numerical_col(dataframe, numerical_col):
    sns.boxplot(x=df[numerical_col])
    plt.show(block=True)

box_numerical_col(df,"Active_Power")

for col in num_cols:
    box_numerical_col(df,col)

df.describe().T

##Degiskenler Arasindaki Korelasyon
###################################
df[num_cols].corr()
df.head()
df.plot.scatter("Active_Power", "windspeed_100m")
plt.show(block=True)

df.plot.scatter("Active_Power", "windgusts_10m")
plt.show(block=True)

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="RdPu")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

df.drop(["apparent_temperature","soil_temperature_0_to_7cm","soil_temperature_7_to_28cm","soil_moisture_7_to_28cm","pressure_msl"],inplace=True,axis=1)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="RdPu")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)


#Zamansal Grafiklerini Inceleyelim
#######################################
df.set_index('DateTime', inplace=True)
#Saatlik

plt.figure(figsize=(15, 5))
plt.title("Günlük Dağıtılan Enerji Zaman Serisi")
plt.plot(df["Active_Power"].resample("H").sum(), linewidth=0.3)
plt.xlabel("Tarih")
plt.ylabel("Aktif Güç")
plt.show(block=True)

#Günlük
plt.figure(figsize=(15, 5))
plt.title("Günlük Dağıtılan Enerji Zaman Serisi")
plt.plot(df["Active_Power"].resample("D").mean(), linewidth=0.5)
plt.xlabel("Tarih")
plt.ylabel("Aktif Güç")
plt.show(block=True)

#Haftalik
plt.figure(figsize=(15, 5))
plt.title("Haftalik Dağıtılan Enerji Zaman Serisi")
plt.plot(df["Active_Power"].resample("W").sum(), linewidth=0.5)
plt.xlabel("Tarih")
plt.ylabel("Aktif Güç")
plt.show(block=True)

#Aylik
plt.figure(figsize=(15, 5))
plt.title("Aylik Dağıtılan Enerji Zaman Serisi")
plt.plot(df["Active_Power"].resample("M").mean(), linewidth=0.5)
plt.xlabel("Tarih")
plt.ylabel("Aktif Güç")
plt.show(block=True)


df.reset_index(inplace=True)
df.head()

#########################################
#Base model
#########################################

df.columns

#Bagimli degiskenimize Logaritmik dönüsüm uyguluyoruz.
df["Active_Power"] = np.log1p(df["Active_Power"].values)

#Bagimli ve bagimsiz degiskeni belirliyoruz.
#Veri setini Train Test olarak ayirdik ama zaman serisi analizí yapacagimiz icin tahmin yapilacak test setini
#tarihe göre siralayarak belli tarihten sonrasi olarak aldik.

X = df.drop(["Active_Power",'Date',"DateTime","Time"], axis = 1)
y = df[["Active_Power"]]


X_train=X.iloc[:16587,:]
X_test=X[16587:]
y_train=y.iloc[:16587,:]
y_test=y[16587:]

y_train.shape, X_train.shape, y_test.shape, X_test.shape
#((18086, 1), (18086, 16), (4522, 1), (4522, 16))

########################
# LightGBM ile Zaman Serisi Base Modeli
########################
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


cols =X_train.columns

lgbtrain = lgb.Dataset(data=X_train, label=y_train, feature_name=cols.tolist())
lgbtest = lgb.Dataset(data=X_test, label=y_test, reference=lgbtrain, feature_name=cols.tolist())

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


model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbtest],
                  num_boost_round=lgb_params['num_boost_round'],
                  feval=lgbm_smape)

y_pred_val = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_val = pd.DataFrame(y_pred_val)
y_pred_vall=np.array(y_pred_val)
y_testt=np.array(y_test)

#SMAPE Hesabi

smape(np.expm1(y_pred_vall), np.expm1(y_testt))
#50.5034 #50.9237 (bazi degiskenleri silince)

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=7, plot=True)

y_pred=np.expm1(y_pred_val)
y_gercek=np.expm1(y_test)

y_pred.to_excel("0809tahminbase.xlsx",index=False)
y_gercek.to_excel("0809gercekbase.xlsx",index=False)

#####################################################
# FEATURE ENGINEERING-TIME SERIES
#####################################################

df=pd.read_excel("Res_Final_R1.xlsx")
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["Active_Power"].replace(0, np.nan, inplace=True)
df.isnull().values.any()
df=df.dropna()
df.shape
#(20734, 20)

df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y-%m-%d %H:%M:%S")


def create_date_features(df):
    df['month'] = df.Date.dt.month
    df['year'] = df.Date.dt.year
    df['day_of_month'] = df.Date.dt.day
    df['day_of_year'] = df.Date.dt.dayofyear
    df['week_of_year'] = df.Date.dt.weekofyear
    df['day_of_week'] = df.Date.dt.dayofweek
    df['is_month_start'] = df.Date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.Date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

########################
# Lag/Shifted Features
########################
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['Power_lag_' + str(lag)] = dataframe['Active_Power'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [1,2,4,24,48,168,8760])
df.head(50)
df.tail()

########################
# Rolling Mean Features
########################

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe['Active_Power'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1, win_type="triang").mean()) + random_noise(dataframe)
    return dataframe

df = roll_mean_features(df, [1,2,12,24,25,48,72])

df.head(24)

########################
# Exponentially Weighted Mean Features
########################
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['active_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe['Active_Power'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.99,0.95, 0.9, 0.8,0.5]
lags = [1,2,4,24,48,72,168]

df = ewm_features(df, alphas, lags)
df.head()
df.info()

########################
# One-Hot Encoding
########################

df = pd.get_dummies(df, columns=['day_of_week', 'month',"year"],drop_first=True)
df.info()

#Bagimli degiskenimize Logaritmik dönüsüm uyguluyoruz.
df["Active_Power"] = np.log1p(df["Active_Power"].values)


#######################
#Train-Test Ayrimi
#######################


X = df.drop(["Active_Power",'Date',"DateTime","Time"], axis = 1)
y = df[["Active_Power"]]
X_train=X.iloc[:16587,:]
X_test=X[16587:]
y_train=y.iloc[:16587,:]
y_test=y[16587:]

#18086
X_test.head()
########################
# LightGBM ile Zaman Serisi Modeli
########################

# LightGBM parameters

lgb_params = {'num_leaves': 30,
              'learning_rate': 0.01,
              'feature_fraction': 0.9,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1500,
              'early_stopping_rounds': 200,
              'nthread': -1}



import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

cols =X_train.columns

lgbtrain = lgb.Dataset(data=X_train, label=y_train, feature_name=cols.tolist())

lgbtest = lgb.Dataset(data=X_test, label=y_test, reference=lgbtrain, feature_name=cols.tolist())


model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbtest],
                  num_boost_round=lgb_params['num_boost_round'],
                  feval=lgbm_smape)


y_pred_val = model.predict(X_test, num_iteration=model.best_iteration)
y_test

y_pred_val = pd.DataFrame(y_pred_val)
y_pred_vall=np.array(y_pred_val)
y_testt=np.array(y_test)

#SMAPE Hesabi

smape(np.expm1(y_pred_vall), np.expm1(y_testt))
#23.9248


def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=7, plot=True)

y_pred=np.expm1(y_pred_val)
y_gercek=np.expm1(y_test)

y_pred.to_excel("0809tahminfeature.xlsx",index=False)



