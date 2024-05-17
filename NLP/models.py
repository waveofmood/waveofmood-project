import pandas as pd
ori_t = pd.read_excel('감성대화말뭉치(최종데이터)_Training.xlsx')
ori_v = pd.read_excel('감성대화말뭉치(최종데이터)_Validation.xlsx')

ori_t.head()
ori_v.head()

ori_t = ori_t.fillna("")
ori_v = ori_v.fillna("")

ori_t["사람문장"] = ori_t["사람문장1"].astype(str)+ori_t["사람문장2"].astype(str)+ori_t["사람문장3"].astype(str)
ori_v["사람문장"] = ori_v["사람문장1"].astype(str)+ori_v["사람문장2"].astype(str)+ori_v["사람문장3"].astype(str)

df_concat = pd.concat([ori_t, ori_v])
df_concat.head()

chatbot = df_concat[["사람문장","감정_대분류"]]

chatbot = chatbot.rename({"감정_대분류": "Emotion"}, axis=1)
chatbot = chatbot.rename({"사람문장": "Sentence"}, axis=1)

chatbot["Emotion"] = chatbot["Emotion"].apply(lambda x:x.strip())

chatbot["Emotion"].value_counts()

chatbot.head()

!pip install mxnet
!pip install gluonnlp pandas tqdm
!pip install sentencepiece
!pip install transformers
!pip install torch

!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

import numpy as np #추가해서 오류 해결 
np.bool = np.bool_
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
device = torch.device("cuda:0")
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# 6개의 감정 class -> 숫자
chatbot.loc[(chatbot['Emotion']=="불안"),'Emotion']= 0 # 불안 => 0
chatbot.loc[(chatbot['Emotion']=="분노"),'Emotion']= 1 # 분노 => 1
chatbot.loc[(chatbot['Emotion']=="상처"),'Emotion']= 2 # 상처 => 2
chatbot.loc[(chatbot['Emotion']=="슬픔"),'Emotion']= 3 # 슬픔 => 3
chatbot.loc[(chatbot['Emotion']=="당황"),'Emotion']= 4 # 당황 => 4
chatbot.loc[(chatbot['Emotion']=="기쁨"),'Emotion']= 5 # 기쁨 => 5

data_list = [ ]
for q, label in zip(chatbot['Sentence'], chatbot['Emotion']):
  data = []
  data.append(q)
  data.append(str(label))

  data_list.append(data)

print(data)
print(data_list[:10])

#train & test 데이터로 나누기
from sklearn.model_selection import train_test_split
dataset_train, dataset_test = train_test_split(data_list, test_size = 0.2, shuffle = True, random_state = 32)

#train & test 데이터로 나누기
from sklearn.model_selection import train_test_split
dataset_train, dataset_test = train_test_split(data_list, test_size = 0.2, shuffle = True, random_state = 32)

# BERTSentenceTransform 클래스 정의
# BERT 모델에 입력으로 넣기 위한 전처리 과정
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset] # 문장 변환
        self.labels = [np.int32(i[label_idx]) for i in dataset] # label 변환

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self): # 전체 데이터셋의 길이 반환
        return (len(self.labels))

# 파라미터 세팅
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-6

# Data tokenization, int encoding, padding

# BERTDataset : 각 데이터가 BERT 모델의 입력으로 들어갈 수 있도록 tokenization, int encoding, padding하는 함수
tok = tokenizer.tokenize # 토크나이저 가져옴

data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, max_len, True, False)

# torch 형식의 dataset을 만들어 입력 데이터셋 전처리 마무리
# 테스트를 수행하기 위한 데이터로 로드
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, num_workers = 5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, num_workers = 5)


# Kobert model 구현

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 6,   # 감정 클래스 수로 조정
                 dr_rate = 0.1,
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes) # 선형 레이어를 통해 BERT의 출력을 감정 클래스로 매핑
        if dr_rate: # 드롭아웃 레이어를 선택적으로 추가
            self.dropout = nn.Dropout(p = dr_rate)
    # 어텐션 마스크 생성: 패딩된 부분 가리기
    # 불필요하게 패딩 토큰에 대해서 어텐션하지 않기
    # 1: 실제 단어이므로 마스킹 하지 않음
    # 0: 패딩 토큰이므로 마스킹한다.
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids): # 예측값 생성
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        # pooler: BERT 모델의 출력 중 하나로, 전체 시퀀스를 요약하는 역할
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict = False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# BERT  모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate = 0.5).to(device)

# optimizer와 schedule 설정
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


optimizer = AdamW(optimizer_grouped_parameters, lr = learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 loss function

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)

# calc_accuracy : 정확도 측정을 위한 함수
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

train_dataloader

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train() # 학습 모드로 설정
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)): # 학습 데이터셋을 순회하면서 각 배치에 대해 반복
        optimizer.zero_grad() #gradient 초기화
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids) # 모델에 데이터를 전달하고 예측 얻기
        loss = loss_fn(out, label) # 손실 계산, 역전파 수행
        loss.backward() # gradient explore 방지
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step() # 모델의 파라미터 업데이트
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label) # 정확도 누적
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    model.eval() # 평가 모드로 설정
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)): # 테스트 데이터셋 순회하면서 각 배치에 대해 반복
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device) # 필요한 데이터를 GPU로 이동
        out = model(token_ids, valid_length, segment_ids) # 모델에 데이터를 전달하고 예측 얻음
        test_acc += calc_accuracy(out, label) # 정확도 누적
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))


# predict 함수 정의

def predict(predict_sentence):  # input = 감정분류하고자 하는 sentence
    data = [predict_sentence, '0'] # input 준비 및 전처리
    dataset_another = [data] # predict sentence를 튜플로 변환

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False) # BERT 모델에 입력으로 사용할 데이터셋 생성: 토큰화한 문장
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)  # torch 형식 변환: mini batch

    model.eval() # 모델 평가 모드

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids) # 모델에 미니배치를 입력으로 주어 예측값을 얻습니다.
        test_eval = []
        for i in out:  # out = model(token_ids, valid_length, segment_ids)
            logits = i
            logits = logits.detach().cpu().numpy()

            # 감정 레이블 및 퍼센트 계산
            emotion_labels = ["불안", "분노", "상처", "슬픔", "당황", "기쁨"]
            probabilities = [np.exp(logit) / np.sum(np.exp(logits)) * 100 for logit in logits]

            for label, percent in zip(emotion_labels, probabilities):
                print(f"{label}: {percent:.2f}%")

            max_index = np.argmax(logits)
            max_emotion_label = emotion_labels[max_index]
            print(f">> 입력하신 내용에서 {max_emotion_label} 느껴집니다.")

# 질문에 0 입력 시 종료
end = 1
while end == 1 :
    sentence = input("하고싶은 말을 입력해주세요 : ")
    if sentence == "0" :
        break
    predict(sentence)
    print("\n")

















