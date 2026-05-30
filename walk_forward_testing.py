from statsmodels.regression.rolling import RollingOLS
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import os
import math
import pandas_ta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models,expected_returns
from pypfopt import objective_functions
from paper_trade_tracker import start_month
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

wikiurl="https://en.wikipedia.org/wiki/NIFTY_500"
CACHE_FILE= "nifty500data.parquet"
PRICE_CACHE = "pricedatadaily.parquet"
outlier_cutoff=0.005
#walk foward testing limits
TEST_START='2024-01-01'

if os.path.exists(CACHE_FILE):
    df=pd.read_parquet(CACHE_FILE)
else:
    headers={
        "User-Agent":"Mozilla/5.0"
    }
    response=requests.get(wikiurl,headers=headers)
    tables=pd.read_html(StringIO(response.text))
    nifty500=tables[4]
    print(nifty500.head())
    nifty500.columns = ['Slno','Company Name','Industry','Symbol' ,'Series' ,'ISIN Code']
    nifty500.columns = ['Slno','Company Name','Industry','Symbol' ,'Series' ,'ISIN Code']
    nifty500=nifty500[nifty500['Symbol']!='Symbol']  # drop header row if present
    nifty500=nifty500[nifty500['Slno']!='Sl.No']     # belt and braces
    symbolslist = nifty500["Symbol"].tolist()
    symbolslist=[s+'.NS' for s in symbolslist]
    print(symbolslist[:5])
    enddate=pd.Timestamp.today()
    startdate=enddate-pd.DateOffset(years=5)
    df=yf.download(tickers=symbolslist,start=startdate,end=enddate).stack(future_stack=True)
    df.index.names=['date','ticker']
    df.columns=df.columns.str.lower()
    df.to_parquet(CACHE_FILE)

df['garman_klass_vol']=((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['close'])-np.log(df['open']))**2)
df['rsi']=df.groupby(level=1)['close'].transform(lambda x:pandas_ta.rsi(close=x,length=20))
df['bb_low']=df.groupby(level=1)['close'].transform(lambda  x: pandas_ta.bbands(close=np.log1p(x),length=20).iloc[:,0])
df['bb_mid']=df.groupby(level=1)['close'].transform(lambda  x: pandas_ta.bbands(close=np.log1p(x),length=20).iloc[:,1])
df['bb_high']=df.groupby(level=1)['close'].transform(lambda  x: pandas_ta.bbands(close=np.log1p(x),length=20).iloc[:,2])
def compute_atr(data):
    high = data['high'].astype(float)
    low = data['low'].astype(float)
    close = data['close'].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    return atr.sub(atr.mean()).div(atr.std())
df['atr']=df.groupby(level=1,group_keys=False).apply(compute_atr)
def compute_macd(close):
    macd=pandas_ta.macd(close=close).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())
df['macd']=df.groupby(level=1,group_keys=False)['close'].apply(compute_macd)
df['rs_vol']=(df['volume']*df['close'])/1e6

lastcols=[c for c in df.columns.unique(0) if c not in ['rs_vol', 'open','volume','high','low','adj close']]
features=df.unstack()[lastcols].resample('ME').last().stack('ticker')
vol=df.unstack('ticker')['rs_vol'].resample('ME').mean().stack('ticker').to_frame('rs_vol')
data=pd.concat([features,vol],axis=1).dropna()

data['rs_vol']=data['rs_vol'].unstack('ticker').rolling(5*12).mean().stack()
data=data.drop(['rs_vol'],axis=1)

def compute_returns(df):
    lags=[1,2,3,6,9,12]
    for lag in lags:
        df[f'return_{lag}m']=df['close'].pct_change(lag).pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),upper=x.quantile(1-outlier_cutoff))).add(1).pow(1/lag).sub(1)
    return df

data=data.groupby(level=1,group_keys=False).apply(compute_returns).dropna()
data=data.drop('close', axis=1) 

factordata=pd.read_csv("FFdata.csv",parse_dates=['Date']).drop('RF',axis=1)
factordata = factordata.set_index('Date')
factordata=factordata.resample('ME').last().div(100)
factordata.index.name='date'
factordata=factordata.join(data['return_1m']).sort_index()
obs=(factordata.groupby(level=1).size())
valid=obs[obs>=10]
factordata=factordata[factordata.index.get_level_values('ticker').isin(valid.index)]

betas=(factordata.groupby(level=1,group_keys=False).apply(lambda x:RollingOLS(endog=x['return_1m'],
       exog=sm.add_constant(x.drop('return_1m',axis=1)),window=min(24,x.shape[0]),
       min_nobs=len(x.columns)+1).fit(params_only=True).params.drop('const',axis=1)))

data=data.join(betas.groupby('ticker').shift())
factors=['SMB','HML','WML','MF']
data.loc[:,factors]=data.groupby('ticker',group_keys=False)[factors].apply(lambda x:x.fillna(x.mean()))

def get_clusters(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df['cluster'] = KMeans(n_clusters=4, random_state=0, init='k-means++').fit(scaled).labels_
    return df
data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)
highest_rsi_cluster = data.groupby(level=0).apply(
    lambda x: x.groupby('cluster')['rsi'].mean().idxmax()
).rename('best_cluster')
data = data.join(highest_rsi_cluster, on='date')
data['is_best'] = data['cluster'] == data['best_cluster']

'''
def plotclusters(data):
    cluster0=data[data['cluster']==0]
    cluster1=data[data['cluster']==1]
    cluster2=data[data['cluster']==2]
    cluster3=data[data['cluster']==3]
    plt.scatter(cluster0.iloc[:,0],cluster0.iloc[:,1],color='red',label="cluster 0")
    plt.scatter(cluster1.iloc[:,0],cluster1.iloc[:,1],color='green',label="cluster 1")
    plt.scatter(cluster2.iloc[:,0],cluster2.iloc[:,1],color='blue',label="cluster 2")
    plt.scatter(cluster3.iloc[:,0],cluster3.iloc[:,1],color='black',label="cluster 3")
    plt.legend()
    plt.show()
    return
plt.style.use('ggplot')
for i in data.index.get_level_values('date').unique().to_list():
    g=data.xs(i,level=0)
    plt.title(f" date{i}")
    plotclusters(g)
'''
filterdf=data[data['is_best']].copy()
filterdf=filterdf.reset_index(level=1)
filterdf.index=filterdf.index+pd.DateOffset(1)
filterdf=filterdf.reset_index().set_index(['date','ticker'])

dates=filterdf.index.get_level_values('date').unique().tolist()
fixeddates={}

for date in dates:
    if(pd.Timestamp(date)>=pd.Timestamp(TEST_START)):
        fixeddates[date]=filterdf.xs(date,level=0).index.to_list()

def optimizeweights(prices):
    if(prices.shape[1]<2):
        return {col:1/prices.shape[1] for col in prices.columns}
    '''
    cols=prices.columns
    returns=np.log(prices).diff().dropna()
    n=prices.shape[1]
    tau_matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            tau_matrix[i][j]=kendalltau(returns[cols[i]],returns[cols[j]])[0]

    pear_matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            pear_matrix[i][j]=math.sin(((math.pi)/2)*tau_matrix[i][j])
    
    vols=returns.std().values
    cov_matrix=pear_matrix*np.outer(vols,vols)
    cov_matrix+=np.eye(n)*1e-8
    cov=pd.DataFrame(cov_matrix,index=cols,columns=cols)
    '''
    returns=expected_returns.mean_historical_return(prices=prices,frequency=252)
    positive_returns_stocks=(returns>0).sum()
    cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    try:
        ef=EfficientFrontier(expected_returns=returns,cov_matrix=cov,weight_bounds=(0,.1),solver='SCS')
        ef.add_objective(objective_functions.L2_reg,gamma=0.5)
        if(positive_returns_stocks>=2):
            ef.max_sharpe()
        else:
            print("Max sharpe optimizing has failed - not enough positive returning stocks")
            ef.min_volatility()
        weights=ef.clean_weights()

    except Exception:
        try:
            ef=EfficientFrontier(expected_returns=returns,cov_matrix=cov,weight_bounds=(0,.1),solver='SCS')
            ef.add_objective(objective_functions.L2_reg,gamma=0.5)
            ef.min_volatility()
            weights=ef.clean_weights()
        except Exception:
            print("Min volatility optimizing has also failed - last resort assigning  equal weights")
            n=prices.shape[1]
            weights={col:1/n for col in prices.columns}

    w = {k: v for k, v in weights.items() if v > 0.01}
    total=sum(w.values())
    w={k:v/total for k,v in w.items()}
    return w

if os.path.exists(PRICE_CACHE):
    newdf = pd.read_parquet(PRICE_CACHE)
else:
    stocks = data.index.get_level_values('ticker').unique().to_list()
    newdf = yf.download(tickers=stocks, start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12))
    newdf.to_parquet(PRICE_CACHE)

returnsdf=np.log(newdf['Close']).diff()
portfoliodf=pd.DataFrame()

first_date=min(fixeddates.keys())
niftyprices=yf.download('^CRSLDX',start=first_date-pd.DateOffset(days=300),end=pd.Timestamp.today())['Close'].squeeze()

def regime_detection(date,prices):
    ma50=prices[date-pd.DateOffset(days=50):date].mean()
    ma200=prices[date-pd.DateOffset(days=200):date].mean()
    if(ma50>ma200): #golden cross
        return True #bull market
    return False #bear market death cross

prev_weights={}
for startdate in fixeddates.keys():
    if(not regime_detection(startdate,niftyprices)):
        continue
    enddate=pd.to_datetime(startdate)+pd.offsets.MonthEnd(0)
    cols=fixeddates[startdate]
    optmizationstartdate=pd.to_datetime(startdate)-pd.DateOffset(months=12)
    optmizationenddate=pd.to_datetime(startdate)-pd.DateOffset(days=1)
    optdf=newdf[optmizationstartdate:optmizationenddate]['Close'][cols]
    threshold = int(0.8 * len(optdf))
    optdf = optdf.dropna(axis=1, thresh=threshold)
    optdf = optdf.ffill().bfill()

    if optdf.shape[1] < 2:
        continue

    w_dict=optimizeweights(prices=optdf)
    
    tempdf=returnsdf[startdate:enddate]
    tempdf=tempdf.stack().to_frame('return').reset_index(level=0)
    tempdf.index.name='ticker'
    tempdf=tempdf.rename(columns={'Date':'date'})
    w = pd.DataFrame(w_dict, index=pd.Series(0)).stack().to_frame('weight')

    all_stocks = set(w_dict.keys()) | set(prev_weights.keys())
    turnover = sum(abs(w_dict.get(s,0) - prev_weights.get(s,0)) for s in all_stocks)
    cost=turnover*0.0022
    prev_weights=w_dict

    w.index=w.index.droplevel(0)
    w.index.name='ticker'
    tempdf=tempdf.join(w)
    tempdf=tempdf.reset_index().set_index(['date','ticker'])
    tempdf['weighted_return']=tempdf['return']*tempdf['weight']
    tempdf.loc[tempdf.index.get_level_values('date')==tempdf.index.get_level_values('date').min(),
                'weighted_return']-=cost/len(tempdf.index.get_level_values('date').unique())
    
    portfoliodf=pd.concat([portfoliodf,tempdf],axis=0)

portfoliodf = portfoliodf.dropna()
portfolioreturns = portfoliodf.groupby(level=0)['weighted_return'].sum().to_frame('strategy_return')


nifty500index = yf.download('^CRSLDX', start=portfolioreturns.index.min(), end=portfolioreturns.index.max())
nifty500returns = np.log(nifty500index['Close']).diff().dropna()
nifty500returns.columns = ['nifty500_return']

latest = portfoliodf.dropna().index.get_level_values('date').max()

#portfolio with the weights
print(portfoliodf.loc[latest].dropna()[['weight']])

daily_returns_from_strat=portfolioreturns['strategy_return']
daily_returns_from_nifty=nifty500returns['nifty500_return']

negative_returns_strat=daily_returns_from_strat[daily_returns_from_strat<0]

strategycurve=daily_returns_from_strat.add(1).cumprod()
niftycurve=daily_returns_from_nifty.add(1).cumprod()

#performace metrics
years=(strategycurve.index.max()-strategycurve.index.min()).days/365.25
runningmax=strategycurve.cummax()
risk_free_returns=0.065 #RBI data not sure tho

cagr=(strategycurve.iloc[-1])**(1/years)-1
volatility=daily_returns_from_strat.std()*np.sqrt(252)
sharpe_ratio=((daily_returns_from_strat.mean()*252)-risk_free_returns)/volatility
drawdown=(strategycurve/runningmax)-1
mdd=drawdown.min()
calmar_ratio=cagr/abs(mdd)
sortino_ratio=((daily_returns_from_strat.mean()*252)-risk_free_returns)/(negative_returns_strat.std()*np.sqrt(252))
metrics={'CAGR':cagr,'Volatility':volatility,'Sharpe ratio':sharpe_ratio,'max drawdown':mdd,
         'calmar ratio':calmar_ratio,'sortino':sortino_ratio}
print(metrics)

plt.figure(figsize=(14,6))
plt.xticks(rotation=45)
ax=plt.gca()
ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%Y'))
plt.plot((strategycurve - 1)*100, label='strategy')
plt.plot((niftycurve - 1)*100, label='NIFTY 500')
plt.legend()
plt.title('strategy vs NIFTY 500')
plt.ylabel('Cumulative Return %')
plt.show()


#paper trade code 

today = niftyprices.index.max()
is_bull = regime_detection(today, niftyprices)
 
latest_weights_dict = (
    portfoliodf.xs(latest, level='date')
    .dropna()[['weight']]
    ['weight']
    .to_dict()
)
 
print("\n" + "="*55)
print(f"  PAPER TRADE — NEXT MONTH")
print(f"  Regime: {'🐂 BULL' if is_bull else '🐻 BEAR'}")
print("="*55)
 
start_month(latest_weights_dict, is_bull)

 
