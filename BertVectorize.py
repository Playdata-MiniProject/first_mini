import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import os

os.chdir('C:/playdata/open/data')

# BERT 모델 및 토크나이저 로드 (사전 훈련된 모델 또는 필요한 모델 선택)
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# CSV 파일에서 키워드 데이터 읽기
csv_file_path = './keybert_result.csv'  # CSV 파일 경로 설정
df = pd.read_csv(csv_file_path, encoding='cp949')

# 키워드 열 선택
keywords_column = 'result'  # CSV 파일에서 키워드가 들어있는 열의 이름

for x in range(1000,len(df),1000):
    # 키워드 데이터를 벡터화
    vectorized_keywords = []

    for keywords in df[keywords_column][x:x+1000]:
        keyword_inputs = tokenizer(keywords, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            keyword_outputs = model(**keyword_inputs)
        keyword_vector = keyword_outputs.last_hidden_state.mean(dim=1)  # 키워드 벡터화 방법을 평균으로 설정
        vectorized_keywords.append(keyword_vector)

    # 벡터화된 키워드를 NumPy 배열로 변환
    vectorized_keywords = torch.cat(vectorized_keywords).numpy()

    # 벡터화된 키워드를 DataFrame에 추가
    df2=pd.DataFrame()
    df2['result'] = [vec.tolist() for vec in vectorized_keywords]

    # 결과를 새로운 CSV 파일로 저장
    output_csv_file_path = './vectorized_keywords.csv'  # 저장할 CSV 파일 경로 설정
    #df2.to_csv(output_csv_file_path, index=False, mode='w')
    df2.to_csv(output_csv_file_path, index=False, mode='a',header=False)
    print(x)

# 결과 출력
#print("벡터화된 키워드 데이터 shape:", vectorized_keywords.shape)
#print("벡터화된 키워드 데이터를 저장한 CSV 파일:", output_csv_file_path)
