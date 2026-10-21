"""Forecast one step ahead using lag features and expanding-window backtesting."""
from kit import *
TITLE='Leakage-Aware Demand Forecasting'
LIVE=False

def design(series,lags=14):
    rows=[];targets=[]
    for t in range(lags,len(series)):
        rows.append(list(series[t-lags:t])+[math.sin(2*math.pi*t/7),math.cos(2*math.pi*t/7)]);targets.append(series[t])
    return np.array(rows),np.array(targets)
def fit(x,y,alpha):
    mean=x.mean(axis=0);ym=y.mean();w=np.linalg.solve((x-mean).T@(x-mean)+alpha*np.eye(x.shape[1]),(x-mean).T@(y-ym))
    return w,float(ym-mean@w)
def run(d,a):
    rng=np.random.default_rng(a.seed);t=np.arange(560);series=50+.03*t+8*np.sin(2*np.pi*t/7)+rng.normal(0,2,560);x,y=design(series)
    validation=[]
    for alpha in (.1,10,100):
        losses=[]
        for end in (200,280,360):
            w,b=fit(x[:end],y[:end],alpha);losses.append(rmse(y[end:end+60],x[end:end+60]@w+b))
        validation.append(dict(alpha=alpha,mean_rmse=float(np.mean(losses))))
    alpha=min(validation,key=lambda z:z['mean_rmse'])['alpha'];cut=len(y)-80;w,b=fit(x[:cut],y[:cut],alpha);p=x[cut:]@w+b;naive=x[cut:,-9]
    return dict(validation=validation,selected_alpha=alpha,test_rmse=rmse(y[cut:],p),seasonal_naive_rmse=rmse(y[cut:],naive),forecast_mode='rolling one-step; observed prior test values allowed'),dict(weights=w,bias=np.array(b),series=series,prediction=p,actual=y[cut:])

if __name__=="__main__":main(run)
