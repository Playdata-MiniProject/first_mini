import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import yfinance as yf

os.chdir('C:/playdata/open/data')


csv_file_path = './sentiment_analysis_VADER2.csv'  # CSV 파일 경로 설정
df2 = pd.read_csv(csv_file_path)

csv_file_path = './NASDAQ_DT_FC_STK_QUT.csv'  # CSV 파일 경로 설정
df = pd.read_csv(csv_file_path, encoding='cp949')
print(df.columns)

# 날짜 데이터를 파싱하고 올바른 날짜 형식으로 저장
df['trd_dt'] = pd.to_datetime(df['trd_dt'], format='%Y%m%d')

result = {}

for stock_name, stock_df in df.groupby('tck_iem_cd'):
    stock_df['Price_Diff'] = stock_df['gts_iem_end_pr'].diff()  # 종가의 차이를 계산하여 가격 변화율 계산
    result[stock_name] = stock_df[['trd_dt', 'gts_iem_end_pr', 'Price_Diff']]


# 결과 출력 (예시로 첫 번째 주식 'AAPL'에 대한 결과만 출력)
print(result['AAPL'])

'''# 가격 변화율을 시각화
plt.figure(figsize=(10, 5))
plt.plot(result['AAPL']['trd_dt'], result['AAPL']['Price_Diff'], label=f'{stock_name} 가격 변화율', marker='o', linestyle='-')
plt.title(f'{stock_name} 주식 가격 변화율')
plt.xlabel('Date')
plt.ylabel('end_price')
plt.legend()
plt.grid(True)

# x-축 눈금 형식 지정
date_format = DateFormatter('%Y-%m-%d')  # 년-월-일 형식
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(prune='both'))  # 눈금 간격 조정

plt.show()'''

# 모든 주식의 가격 변화율 평균 계산
average_price_diff = {}
for stock_name, stock_data in result.items():
    average_price_diff[stock_name] = stock_data['Price_Diff'].mean()

# 평균 가격 변화율을 정렬
sorted_average_price_diff = dict(sorted(average_price_diff.items(), key=lambda item: item[1], reverse=True))

# 평균 가격 변화율을 시각화
plt.figure(figsize=(10, 5))
plt.bar(sorted_average_price_diff.keys(), sorted_average_price_diff.values())
plt.title('모든 주식의 평균 가격 변화율')
plt.xlabel('주식 이름')
plt.ylabel('평균 가격 변화율')
plt.grid(axis='y')

plt.show()

# 이상치를 제거 (상위 20%와 하위 20%)
total_stocks = len(sorted_average_price_diff)
top_cutoff = int(total_stocks * 0.2)
bottom_cutoff = int(total_stocks * 0.8)
trimmed_average_price_diff = dict(list(sorted_average_price_diff.items())[top_cutoff:bottom_cutoff])

# 이상치가 제거된 평균 가격 변화율을 시각화
plt.figure(figsize=(10, 5))
plt.bar(trimmed_average_price_diff.keys(), trimmed_average_price_diff.values())
plt.title('모든 주식의 평균 가격 변화율 (이상치 제거)')
plt.xlabel('주식 이름')
plt.ylabel('평균 가격 변화율')
plt.grid(axis='y')

plt.show()

# 음수 가격 변화율을 제거
filtered_average_price_diff = {k: v for k, v in trimmed_average_price_diff.items() if v >= 0}

print(len(filtered_average_price_diff.keys()))

positive_stock=filtered_average_price_diff.keys()

'''combined_df = pd.concat([df2, df3], ignore_index=True)  # ignore_index=True로 설정하면 인덱스를 재설정합니다.

# 합쳐진 데이터프레임을 CSV 파일로 저장
combined_df.to_csv('./Summary_Sentiment_total.csv', index=False)  # index=False로 설정하면 인덱스를 저장하지 않습니다.
'''

# 감성 분석 값이 긍정적인 주식들을 찾아내기
positive_sentiment_stocks = df2[df2['sentiment'] >= 0.75]['tck_iem_cd']

# 긍정적인 감성 분석 값을 가진 주식들과 이상치 제거한 주식 데이터의 교집합 찾기
common_stocks = set(positive_sentiment_stocks).intersection(set(positive_stock))

print('positive_sentiment_stocks :',len(positive_sentiment_stocks))
print('common_stocks',len(common_stocks))
print(common_stocks)

stock_codes = list(common_stocks)

'''# 시가총액이 1000억 달러 이상인 주식만 추출
selected_stocks = []

for stock_code in stock_codes:
    try:
        stock_info = yf.Ticker(stock_code).info
        market_cap = stock_info.get('marketCap')

        if market_cap is not None and market_cap >= 100000000000:  # 1000억 이상 (1,000,000,000,000)
            selected_stocks.append(stock_code)
    except:
        continue

print("선택된 주식 코드:", selected_stocks)'''

# 주식 코드와 시가총액을 저장할 딕셔너리
stock_market_caps = {}

# 시가총액 정보 수집
for stock_code in stock_codes:
    try:
        stock_info = yf.Ticker(stock_code).info
        market_cap = stock_info.get('marketCap')

        if market_cap is not None:
            stock_market_caps[stock_code] = market_cap
    except:
        continue

# 시가총액이 가장 높은 5개 주식 추출
selected_stocks = sorted(stock_market_caps, key=lambda k: stock_market_caps[k], reverse=True)[:5]

print("시가총액이 가장 높은 5개 주식 코드:", selected_stocks)

