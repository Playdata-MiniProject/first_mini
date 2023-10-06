import pandas as pd
import os

os.chdir('C:/playdata/open/data')

# 주식 2700개 정보가 있는 CSV 파일 경로
stocks_csv_path = './NASDAQ_FC_STK_IEM_IFO.csv'
# 원-핫 인코딩 결과 파일 경로
one_hot_csv_path = './priceDiff_one_hot_encoding.csv'  # 원-핫 인코딩 결과 파일 경로

# 주식 정보 CSV 파일 불러오기
stocks_df = pd.read_csv(stocks_csv_path, encoding='cp949')

# 원-핫 인코딩 결과 CSV 파일 불러오기
one_hot_df = pd.read_csv(one_hot_csv_path, encoding='cp949')

# '주식 코드' 컬럼 값에 공백 제거
stocks_df['tck_iem_cd'] = stocks_df['tck_iem_cd'].str.strip()
one_hot_df['tck_iem_cd'] = one_hot_df['tck_iem_cd'].str.strip()

# 주식 정보와 원-핫 인코딩 결과를 병합 (주식 코드를 기준으로 병합)
merged_df = pd.merge(stocks_df, one_hot_df, how='left')

# 필요 없는 열(column) 삭제 (원핫인코딩 결과의 주식 이름 열은 삭제)
#merged_df = merged_df.drop(columns=['tck_iem_cd'])


# 병합된 데이터프레임을 새로운 CSV 파일로 저장
merged_csv_path = './NASDAQ_FC_STK_IEM_IFO_one_hot.csv'
merged_df.to_csv(merged_csv_path, index=False, encoding='utf-8-sig', mode='w')
