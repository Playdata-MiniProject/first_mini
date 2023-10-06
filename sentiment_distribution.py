import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('C:/playdata/open/data')

# CSV 파일 경로 설정 (감성분석 결과를 저장한 CSV 파일)
csv_file_path = 'Summary_Sentiment_total.csv'  # 파일 경로를 적절하게 수정하세요.

# CSV 파일 읽기
df = pd.read_csv(csv_file_path)

# 감성분석 결과 컬럼 선택 (예시: 'sentiment' 컬럼)
sentiment_values = df['sentiment'].tolist()

# 히스토그램으로 시각화
plt.hist(sentiment_values, bins=20, color='skyblue', alpha=0.7)
plt.title('감성분석 결과 분포')
plt.xlabel('감성분석 결과')
plt.ylabel('빈도')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
