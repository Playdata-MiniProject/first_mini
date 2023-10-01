import pandas as pd
import os
from keybert import KeyBERT

os.chdir('C:/Users/User/Downloads/open')

# CSV 파일 경로
csv_file_path = './RSS_total.csv'

# CSV 파일을 pandas DataFrame으로 읽기
df = pd.read_csv(csv_file_path,encoding='cp949')

# KeyBERT 모델 로드
model = KeyBERT('distilbert-base-nli-mean-tokens')

# 키워드 추출 함수 정의
def extract_keywords(text, num_keywords=5):
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', use_maxsum=True, nr_candidates=20)
    return [keyword[0] for keyword in keywords[:num_keywords]]

# 각 행에 대한 키워드 추출 및 결과 저장
df['Keywords'] = df['news_smy_ifo'][0:1000].apply(extract_keywords)

# 결과를 CSV 파일로 저장
df.to_csv('./output_keywords.csv', index=False)
