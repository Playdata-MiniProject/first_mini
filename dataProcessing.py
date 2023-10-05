import os
import pandas as pd

os.chdir('C:\playdata\open\data')

#df=pd.read_csv('./NASDAQ_FC_STK_IEM_IFO.csv', encoding='cp949')
#df=pd.read_csv('./NASDAQ_DT_FC_STK_QUT.csv', encoding='cp949')
for x in range(1,9):
    df = pd.read_csv('./NASDAQ_RSS_IFO/NASDAQ_RSS_IFO_20230{}.csv'.format(x), encoding='cp949')
    missing_columns = df.isnull().sum()
    print("열별 결측치 개수:\n", missing_columns)

    # 행별 결측치 개수 확인
    missing_rows = df.isnull().sum(axis=1)
    print("행별 결측치 개수:\n", missing_rows)
