import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer  # 어텐션 레이어 관련 모듈
from .embed import DataEmbedding, TokenEmbedding  # 데이터 임베딩 관련 모듈


# EncoderLayer: 트랜스포머 인코더의 하나의 층을 정의 (어텐션 + 피드포워드 네트워크)
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # 피드포워드 네트워크의 차원 설정 (기본값: d_model의 4배)
        self.attention = attention  # 어텐션 레이어 (AnomalyAttention 기반)
        # 1D 컨볼루션 레이어 (피드포워드 네트워크)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout 적용
        self.dropout = nn.Dropout(dropout)
        # 활성화 함수 (ReLU 또는 GELU 선택 가능)
        self.activation = F.relu if activation == "relu" else F.gelu

    # x: 입력 텐서, attn_mask: 어텐션 마스크
    def forward(self, x, attn_mask=None):
        # 어텐션을 통해 새로운 x와 어텐션 관련 정보 반환
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # 어텐션 결과와 원본 x를 더하고 드롭아웃 적용
        x = x + self.dropout(new_x)
        # LayerNorm 적용
        y = x = self.norm1(x)
        # 피드포워드 네트워크 (컨볼루션 적용 후 활성화 함수)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # 다시 원래 차원으로 복원

        # 최종 출력과 함께 LayerNorm 적용, 어텐션 관련 정보도 반환
        return self.norm2(x + y), attn, mask, sigma


# Encoder: 여러 개의 EncoderLayer를 쌓아서 인코더 구성
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # 어텐션 레이어 리스트
        self.norm = norm_layer  # 마지막에 적용할 LayerNorm (선택적)

    # x: 입력 텐서, attn_mask: 어텐션 마스크
    def forward(self, x, attn_mask=None):
        # 어텐션 레이어에서 사용할 시리즈, 우선순위, 시그마 리스트 초기화
        series_list = []
        prior_list = []
        sigma_list = []
        # 모든 어텐션 레이어를 순차적으로 적용
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            # 어텐션 관련 정보 저장
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        # 만약 마지막에 정규화 레이어가 있으면 적용
        if self.norm is not None:
            x = self.norm(x)

        # 인코더 출력 및 어텐션 관련 정보를 반환
        return x, series_list, prior_list, sigma_list


# AnomalyTransformer: 이상 탐지에 특화된 트랜스포머 모델 정의
class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention  # 어텐션 정보 출력 여부

        # 임베딩 레이어 설정 (DataEmbedding 사용)
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder 설정: e_layers 만큼의 EncoderLayer를 쌓음
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),  # AnomalyAttention 기반의 어텐션 레이어
                    d_model,  # d_model: 임베딩 차원
                    d_ff,  # 피드포워드 차원
                    dropout=dropout,  # 드롭아웃 비율
                    activation=activation  # 활성화 함수 선택 (GELU)
                ) for l in range(e_layers)  # e_layers 만큼의 인코더 레이어 생성
            ],
            norm_layer=torch.nn.LayerNorm(d_model)  # 마지막에 적용할 정규화 레이어
        )

        # 최종 예측을 위한 Linear 레이어
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        # 입력 데이터를 임베딩
        enc_out = self.embedding(x)
        # 임베딩된 데이터를 인코더에 전달하여 어텐션 수행
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        # 인코더 결과를 기반으로 최종 예측값 생성
        enc_out = self.projection(enc_out)

        # 어텐션 정보 출력 여부에 따라 결과 반환
        if self.output_attention:
            return enc_out, series, prior, sigmas  # 예측값과 함께 어텐션 정보도 반환
        else:
            return enc_out  # 예측값만 반환
