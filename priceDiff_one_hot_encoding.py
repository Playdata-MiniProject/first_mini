import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir('C:/playdata/open/data')

# CSV 파일 경로 설정
csv_file_path = './NASDAQ_DT_FC_STK_QUT.csv'
df = pd.read_csv(csv_file_path, encoding='cp949')

result = {}

for stock_name, stock_df in df.groupby('tck_iem_cd'):
    stock_df['Price_Diff'] = stock_df['gts_iem_end_pr'].diff()  # 종가의 차이를 계산하여 가격 변화율 계산
    result[stock_name] = stock_df[['trd_dt', 'gts_iem_end_pr', 'Price_Diff']]

# 모든 주식의 가격 변화율 평균 계산
average_price_diff = {}
for stock_name, stock_data in result.items():
    average_price_diff[stock_name] = stock_data['Price_Diff'].mean()

# 평균 가격 변화율을 정렬
sorted_average_price_diff = dict(sorted(average_price_diff.items(), key=lambda item: item[1], reverse=True))

# 상위 20%, 하위 20% 및 나머지 60%로 나누어 원-핫 인코딩
total_stocks = len(sorted_average_price_diff)
top_cutoff = int(total_stocks * 0.2)
bottom_cutoff = int(total_stocks * 0.8)
one_hot_encoding = {}

for i, (stock_name, price_diff) in enumerate(sorted_average_price_diff.items()):
    if i < top_cutoff:
        one_hot_encoding[stock_name] = 0
    elif i < bottom_cutoff:
        one_hot_encoding[stock_name] = 1
    else:
        one_hot_encoding[stock_name] = 2

# 결과를 DataFrame으로 변환
one_hot_df = pd.DataFrame(list(one_hot_encoding.items()), columns=['tck_iem_cd', 'price_diff_range_20'])

# 결과를 CSV 파일로 저장
one_hot_df.to_csv('./priceDiff_one_hot_encoding.csv', index=False, mode='w')
