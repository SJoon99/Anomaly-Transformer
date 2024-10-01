import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


# PositionalEmbedding: 입력 데이터의 순차적 위치 정보를 임베딩하여 Transformer가 순서를 학습할 수 있도록 함
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 위치 임베딩을 미리 계산하여 저장
        pe = torch.zeros(max_len, d_model).float()  # max_len x d_model 크기의 텐서를 생성
        pe.require_grad = False  # 학습 중에 이 값이 업데이트되지 않도록 설정

        # 위치 정보를 나타내는 텐서를 생성 (0부터 max_len까지)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # d_model 차원에서 짝수와 홀수 인덱스에 대해 다르게 처리할 수 있도록 분할
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 짝수 인덱스: sin 함수 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 인덱스: cos 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 배치 차원을 추가해줌
        # 미리 계산된 위치 임베딩을 register_buffer로 등록하여 추론 중에는 사용 가능하지만 학습 중에는 업데이트되지 않도록 함
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 데이터 x의 시퀀스 길이에 맞게 위치 임베딩을 반환
        return self.pe[:, :x.size(1)]


# TokenEmbedding: 입력 데이터를 고차원 벡터로 변환하는 임베딩 레이어
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # PyTorch 버전에 따른 패딩 값 설정
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # Conv1d 레이어를 사용하여 입력 시퀀스를 고차원 공간으로 변환
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # Conv1d 레이어의 가중치를 He 초기화 방식으로 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # 입력 데이터의 차원을 (batch_size, feature_size, sequence_length)로 맞춰줌
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# DataEmbedding: 값 임베딩과 위치 임베딩을 결합한 최종 임베딩 레이어
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        # 입력 데이터를 고차원 벡터로 변환하는 TokenEmbedding 레이어
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 위치 정보를 임베딩하는 PositionalEmbedding 레이어
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # 드롭아웃을 사용하여 과적합을 방지
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 값 임베딩과 위치 임베딩을 더한 후 드롭아웃을 적용하여 반환
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
