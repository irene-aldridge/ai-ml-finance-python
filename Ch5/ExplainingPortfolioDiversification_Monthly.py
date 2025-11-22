import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
outpath = 'C:/Users/ABlE/Documents/Books/FinEng/Results/Ch5/'
inpath = 'C:/Users/ABlE/Documents/AbleAlpha/Data/Monthly/YF/'

ticker1 = 'MORN'
ticker2 = 'BUD'

df1 = pd.read_csv(inpath + ticker1 + '_Monthly.csv', skiprows=[1,2])
df2 = pd.read_csv(inpath + ticker2 + '_Monthly.csv', skiprows=[1,2])

df1 = df1[df1['Price'] > '2015-01-01'].reset_index()
df2 = df2[df2['Price'] > '2015-01-01'].reset_index()
df1['Close1'] = df1['Close']
df2['Close2'] = df2['Close']

df = df1[['Price', 'Close1']].merge(df2[['Price', 'Close2']], how='inner',left_on='Price', right_on='Price').reset_index()
c = np.corrcoef(df['Close1'], df['Close2'])
print('correl of prices = ', c)

plt.plot(df['Close1'], label=ticker1)
plt.plot(df['Close2'], label=ticker2)
xticks = np.arange(0, len(df), 24)
plt.xticks(xticks, df['Price'][xticks], rotation=90)
plt.xlabel('Dates')
plt.title('Prices of '+ticker1+' and '+ticker2)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(outpath + 'prices_'+ticker1+'_'+ticker2+'.png')
plt.show()

df['R1'] = (df['Close1']/df['Close1'].shift(1)-1).fillna(0)
df['R2'] = (df['Close2']/df['Close2'].shift(1)-1).fillna(0)
# 
c = np.corrcoef(df['R1'], df['R2'])
print('return correlation = ',c)

plt.plot(df['R1'], label=ticker1)
plt.plot(df['R2'], label=ticker2)
xticks = np.arange(0, len(df), 24)
plt.xticks(xticks, df['Price'][xticks], rotation=90)
plt.xlabel('Dates')
plt.title('Returns of '+ticker1+' and '+ticker2 )
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(outpath + 'returns_'+ticker1+'_'+ticker2+'_monthly.png')
plt.show()

x1 = (np.var(df['R2']) - np.cov(df['R1'],df['R2'])[0,1])/(np.var(df['R1'])+np.var(df['R2']) - np.cov(df['R1'],df['R2'])[0,1])
print('x1 = ', x1)

plt.scatter(df['R1'], df['R2'], s=2)
plt.xlabel('Returns '+ticker1)
plt.ylabel('Returns ' + ticker2)
plt.title('Returns of '+ticker1+' and '+ticker2 )
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(outpath + 'scatter_returns_'+ticker1+'_'+ticker2+'_monthly.png')
plt.show()


df['S0.25'] = 0.25*df['Close1']/(df['Close1'][0]) + 0.75*df['Close2']/(df['Close2'][0])
df['S0.5'] = 0.5*df['Close1']/(df['Close1'][0]) + 0.5*df['Close2']/(df['Close2'][0])
df['S0.75'] = 0.75*df['Close1']/(df['Close1'][0]) + 0.25*df['Close2']/(df['Close2'][0])


plt.plot(df['S0.25'], label='x1 = 0.25')
plt.plot(df['S0.5'], label='x1 = 0.5')
plt.plot(df['S0.75'], label='x1 = 0.75')
xticks = np.arange(0, len(df), 24)
plt.xticks(xticks, df['Price'][xticks], rotation=90)
plt.xlabel('Dates')
plt.title('Portfolios of '+ticker1 + ' and '+ticker2)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(outpath + 'chart_portfolio_'+ticker1+'_'+ticker2+'_monthly.png')
plt.show()

pmeans = {}
pstds = {}
psharps = {}

for x in range(100):
    x1 = x/100
    df['Rcombo'] = x1*df['R1'] + (1-x1)*df['R2']
    
    pmeans[x1] = np.mean(df['Rcombo'])*251
    pstds[x1] = np.std(df['Rcombo'])*np.sqrt(251)
    psharps[x1] = np.mean(df['Rcombo'])/np.std(df['Rcombo'])*np.sqrt(251)
    if((x1 == 0.25)|(x1 == 0.5)|(x1 == 0.75)):
        print(x1, pmeans[x1], pstds[x1], psharps[x1])


#print('x1 = ', x1)
#print('pmeans = ', pmeans)
#print('pstds =', pstds)
#print('psharps =', psharps)

plt.plot(pmeans.keys(), pmeans.values(), 'k', linewidth=4, label='portfolio means')
plt.plot(pmeans.keys(), pstds.values(), 'k--', label='portfolio volatility')
plt.plot(pmeans.keys(), psharps.values(), label='portfolio Sharpe')
plt.xlabel('x1')
plt.title('Portfolio Characteristics of '+ticker1+' and '+ticker2)
plt.legend()
plt.tight_layout()
plt.savefig(outpath + 'portfolio_charateristics.png')
plt.show()


#p1 = [df['Close'][0]]
p1 = []
s = []
for index, row in df.iterrows():
    if(len(p1) == 0):
        p1.append(row['Close'])
    else:
        if(np.random.normal(0,1) > 0):
            k = row['R1']
        else:
            k = np.random.normal(0,0.05)
        p1.append(p1[-1]*(1+k))
    s = row['Close']+p1[-1]
    
c = np.corrcoef(df['Close'], p1)
print('corr = ',c)

plt.plot(df['Close'])
plt.plot(p1)
plt.title('A sample plot of two time series')
xticks = np.arange(0, len(df), 250)
plt.xticks(xticks, df['Price'][xticks], rotation=90)
plt.tight_layout()
plt.savefig(outpath + 'chart_corr_'+str(c)+'.png')
plt.show()


plt.plot(s)
plt.title('The equal-part sum of the two time series')
xticks = np.arange(0, len(df), 250)
plt.xticks(xticks, df['Price'][xticks], rotation=90)
plt.tight_layout()
plt.savefig(outpath + 'chart_corr_'+str(c)+'.png')
plt.show()

