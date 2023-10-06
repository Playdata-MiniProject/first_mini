import os
import torch
import pickle
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# 미리 훈련된 BERT 모델 및 토크나이저 로드
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# BERT 모델의 파라미터를 동결하여 학습되지 않도록 설정
# BERT 모델의 가중치가 무작위로 초기화되지 않도록 설정
for param in model.bert.parameters():
    param.requires_grad = False



# # 벡터화된 데이터 로드 (pickle 파일)
# with open('./data/vectors.pickle', 'rb') as file:
#     vectorized_data = pickle.load(file)

rss_info = pd.read_csv('./data/RSS_total.csv', encoding='ANSI')

print(rss_info[:15077])

print(rss_info.shape)

# 벡터화된 데이터에서 텍스트 추출
text = rss_info.get('text', '')

# Fine-tuning을 위한 학습 데이터 생성
# 이 부분은 실제 학습 데이터를 생성하는 부분으로, 주식 관련 뉴스 데이터와 레이블을 사용하여 데이터를 준비해야 합니다.
inputs = tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
labels = torch.tensor([1]).unsqueeze(0)  # 긍정 레이블 (1)을 배치 차원 추가

# DataLoader를 사용하여 데이터 로드
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=32)

# 옵티마이저 설정
AdamW.no_deprecation_warning = True
optimizer = AdamW(model.parameters(), lr=1e-5)

# Fine-tuning 반복 루프
num_epochs = 10  # Fine-tuning을 몇 번 반복할지 설정
for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    total_loss = 0.0
    
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 모델 순전파
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 에포크마다 손실 출력
    print(f"Epoch {epoch + 1} / {num_epochs}, Loss: {total_loss / len(dataloader)}")



# Fine-tuned 모델을 저장할 디렉터리 경로 설정
save_directory = 'fine_tuned_models'

# 디렉토리가 없으면 생성
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Fine-tuned 모델 저장
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)



# 모델을 추론 모드로 설정
model.eval()

# 감성 분석 수행
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# 로짓을 확률로 변환 (소프트맥스)
probs = torch.softmax(logits, dim=1)
positive_probability = probs[0][1].item()  # 긍정 클래스의 확률 값

# 감성 분석 결과 출력
if positive_probability > 0.5:
    print("긍정적인 감정입니다.")
else:
    print("부정적인 감정입니다.")