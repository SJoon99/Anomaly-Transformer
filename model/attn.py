import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


# TriangularCausalMask: 트랜스포머에서 자기회귀 모델의 특성상 과거 데이터만을 참조하도록 하는 마스크 생성
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        # B: 배치 크기, L: 시퀀스 길이
        mask_shape = [B, 1, L, L]  # (B, 1, L, L) 크기의 마스크 텐서
        # 상삼각 행렬 형태의 마스크를 생성하여 미래 시점의 값을 가리지 않도록 만듦
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask  # 생성된 마스크를 반환


# AnomalyAttention: 이상 탐지에 사용되는 특화된 어텐션 레이어
class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale  # 어텐션 점수에 사용할 스케일링 값
        self.mask_flag = mask_flag  # 마스크 적용 여부
        self.output_attention = output_attention  # 어텐션 출력 여부
        self.dropout = nn.Dropout(attention_dropout)  # 어텐션 결과에 드롭아웃 적용
        window_size = win_size  # 윈도우 크기 설정
        # 윈도우 내의 시퀀스 간의 거리를 저장할 distance 행렬을 초기화
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)  # 각 위치 간의 절대적인 거리를 기록

    # 어텐션 계산을 위한 forward 함수
    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape  # queries의 배치 크기, 시퀀스 길이, 헤드 수, 임베딩 차원
        _, S, _, D = values.shape  # values의 시퀀스 길이 및 다른 차원
        scale = self.scale or 1. / sqrt(E)  # 스케일링 값 설정, 없으면 기본 sqrt(E)의 역수 사용

        # 쿼리와 키 간의 내적을 통해 어텐션 점수 계산 (batch, head, length, length)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 마스크가 필요한 경우 TriangularCausalMask 적용
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)  # 마스크가 없다면 생성
            scores.masked_fill_(attn_mask.mask, -np.inf)  # 마스크로 가려진 부분을 음의 무한대로 채움

        attn = scale * scores  # 스케일링을 적용한 어텐션 점수 계산

        # sigma 값을 시퀀스 간 거리 기반으로 조정
        sigma = sigma.transpose(1, 2)  # (B, L, H) -> (B, H, L)
        window_size = attn.shape[-1]  # 어텐션 창 크기
        sigma = torch.sigmoid(sigma * 5) + 1e-5  # sigma 값을 sigmoid로 조정
        sigma = torch.pow(3, sigma) - 1  # sigma 값을 지수로 변환
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # sigma 크기 조정 (B, H, L, L)

        # 시퀀스 간 거리에 기반한 우선순위 계산
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))  # 가우시안 분포 기반 우선순위

        series = self.dropout(torch.softmax(attn, dim=-1))  # softmax를 적용하여 어텐션 가중치 계산 후 드롭아웃
        V = torch.einsum("bhls,bshd->blhd", series, values)  # 어텐션 가중치를 values에 곱해 최종 결과 계산

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)  # 어텐션 가중치와 추가 정보를 반환
        else:
            return (V.contiguous(), None)  # 어텐션 가중치 없이 결과만 반환


# AttentionLayer: 일반적인 어텐션 레이어, n_heads는 다중 헤드 어텐션을 위한 값
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        # 기본적으로 n_heads로 나눈 차원을 사용
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # 입력을 정규화하기 위한 LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # 내부 어텐션 모듈 (AnomalyAttention 같은 커스텀 어텐션)
        self.inner_attention = attention

        # 쿼리, 키, 값, 시그마를 각각 투영하는 레이어
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # d_model -> (d_keys * n_heads)로 차원 변환
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)  # 시그마 값 생성

        # 출력 projection (어텐션 결과를 다시 d_model 크기로 투영)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads  # 다중 헤드 수 설정

    # 쿼리, 키, 값 입력에 대한 어텐션 계산
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape  # 배치 크기, 쿼리의 시퀀스 길이
        _, S, _ = keys.shape  # 키의 시퀀스 길이
        H = self.n_heads  # 다중 헤드 수
        x = queries  # 입력된 쿼리 데이터

        # 쿼리, 키, 값에 대해 각각 프로젝션을 수행해 차원을 맞춤
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)  # sigma 값도 투영하여 계산

        # 내부 어텐션 모듈 (AnomalyAttention)을 호출하여 어텐션 계산
        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)  # 결과를 다시 (B, L, d_model)로 reshape

        return self.out_projection(out), series, prior, sigma  # 최종 출력 및 추가 정보를 반환
