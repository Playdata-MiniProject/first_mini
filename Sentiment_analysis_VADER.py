import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

os.chdir('C:/playdata/open/data')

# CSV 파일 경로 설정 (해외뉴스 데이터 파일)
csv_file_path = './RSS_total.csv'

# CSV 파일 읽기
df = pd.read_csv(csv_file_path, encoding='cp949')

# VADER 감성 분석기 초기화
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

'''# 모델과 데이터를 GPU로 이동
model.to(device)'''

# 감성분석을 위한 함수 정의
def perform_sentiment_analysis(text):
    # VADER를 사용하여 감성분석 수행
    sentiment_scores = analyzer.polarity_scores(text)

    # 'compound' 점수를 -1에서 1 사이의 값으로 정규화
    sentiment_score = sentiment_scores['compound']

    return sentiment_score


# 데이터프레임에 새로운 감성분석 결과 열 추가
for x in range(0,len(df),1000):

    df2 = pd.DataFrame()
    df2['sentiment'] = df['news_smy_ifo'][x:x+1000].apply(perform_sentiment_analysis)

    # 결과를 CSV 파일로 저장 (새로운 감성분석 결과가 있는 열이 추가된 CSV 파일)
    output_csv_file_path = './sentiment_analysis_VADER.csv'
    df2.to_csv(output_csv_file_path, index=False, mode='a', header=False)
    print("complete cell :", x+1000)
