import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import one_hot_encoding_prediction

class RNN(nn.Module):
    def __init__(self, dim_input, dim_recurrent, dim_output):
        super(RNN, self).__init__()

        """
        dim_input: C
        dim_recurrent: D
        dim_output: K
        """

        self.x2h = nn.Linear(dim_input, dim_recurrent)
        self.h2h = nn.Linear(dim_recurrent, dim_recurrent, bias=True)
        self.h2y = nn.Linear(dim_recurrent, dim_output)
        self.relu = nn.ReLU()

        nn.init.xavier_normal_(self.x2h.weight)
        nn.init.xavier_normal_(self.h2h.weight)
        nn.init.xavier_normal_(self.h2y.weight)

    def forward(self, x, h_t=None):

        """
        x: shape = (T, N, C)
        W_x: shape = (C, D)
        초기 h: shape = (1, N, D)
        W_h: shape = (D, D)

        => x X W_x: (T, N, C) X (C, D) = (T, N, D)
           (초기 h) X W_h: (1, N, D) X (D, D) = (1, N, D)  
           
           h: (T, N, D) + (1, N, D) = (T, N, D) broadcasting

        w_y: shape = (D, K)

        y: h X w_y
         = (T, N, D) X (D, K) = (T, N, K)  

        y: shape = (T, N, K)
        h: shape = (T, N, D)  
        """
         
        N = x.shape[1]
        D = self.h2h.weight.shape[0]

        # 초기 hidden state를 (1, N, D) shape의 0텐서로 설정
        if h_t is None:
            h_t = torch.zeros(1, N, D, dtype=torch.float32)

        h = []

        for x_t in x:
            h_t = self.x2h(x_t.unsqueeze(0)) + self.h2h(h_t)
            h_t = self.relu(h_t)
            h.append(h_t)

        # 배열 h에 저장된 모든 hidden state들을 dim=0 방향으로 합치기
        all_h = torch.cat(h, dim=0)

        all_y = self.h2y(all_h)

        return all_y, all_h       
    



class AttentionSeq2Seq(nn.Module):
    def __init__(self, dim_input, dim_recurrent, dim_output):
        super(AttentionSeq2Seq, self).__init__()

        """
        dim_input: 입력 데이터 차원(C)
        dim_recurrent: hidden state 차원(D)
        dim_output: 디코더 출력 차원(K)
        """

        self.encoder = RNN(dim_input, dim_recurrent, dim_output)
        self.decoder = RNN(dim_input, dim_recurrent, dim_output)

        """
        Attention 메커니즘을 적용하기 위해 dense layer를 추가로 생성
        W_alpha: 인코더의 hidden state들에 가중치를 부여해주기 위해 생성
        concat_dense: 이후에 디코더의 y_t와 Context vector c_t를 concatenate 해주기 위해 생성
        """
        self.W_alpha = nn.Linear(dim_recurrent, dim_recurrent, bias=False)
        self.concat_dense = nn.Linear(dim_recurrent + dim_output, dim_output)

        nn.init.xavier_normal_(self.W_alpha.weight)
        nn.init.xavier_normal_(self.concat_dense.weight)


    def forward(self, x):
        T, N, C = x.shape

        y = []

        # 인코더를 통해 hidden state를 받음, 인코더의 출력은 취급하지 않음
        _, enc_h = self.encoder(x)

        # 마지막 step에서의 hidden state를 지정
        h_t = enc_h[-1:]

         # <sos> 토큰 즉, 디코더 첫 step의 입력을 나타내는 start 토큰 설정
        s_t = torch.zeros(1, N, C)
        s_t[:, :, -2] = 1

        # 인코더의 hidden state들을 W_alpha layer에 통과시켜 Attention Score의 일부를 미리 계산
        precomputed_encoder_score_vectors = self.W_alpha(enc_h)

        # 각 디코더 step에서의 Attention Weight를 저장하기 위해 생성
        a_ij = []

        # 타임스텝 T 동안 디코더 반복 실행
        for _ in range(T):

            # 우선 step을 진행하기 전에 현재 hidden state에 대한 Attention Score 계산
            e_t = (precomputed_encoder_score_vectors*h_t).sum(dim=-1)

            # 위에서 구한 Attention Score를 softmax 함수에 통과시켜 Attention Weight 도출
            a_t = F.softmax(e_t, dim=0)

            # Attention Weight를 리스트 a_ij에 저장
            a_ij.append(a_t.unsqueeze(0))

            # 인코더의 hidden state들과 Attention Weight를 통해 Context Vector 도출
            c_t = (a_t[..., None]*enc_h).sum(dim=0, keepdims=True)

            # 하나의 디코더 step을 진행(s_t, h_t 입력)하여 y_t와 h_t를 받음
            y_t, h_t = self.decoder(s_t, h_t)

            # 도출된 y_t와 이전에 구한 Context Vector를 dim=-1 방향으로 concatenate
            y_and_c = torch.cat([y_t, c_t], dim=-1)

            # concatenate된 벡터를 concat_dense layer에 통과시켜 새로운 y_t를 도출한 후 y에 저장
            y_t = self.concat_dense(y_and_c)
            y.append(y_t)

            # 디코더의 다음 step의 입력값을 새로 할당
            s_t = one_hot_encoding_prediction(y_t)

        # 리스트 y에 저장된 모든 y들을 dim=0 방향으로 합치기
        y = torch.cat(y, dim=0)

        # 리스트 a_ij에 저장된 모든 Attention Weight들을 dim=0 방향으로 합치기
        a_ij = torch.cat(a_ij, dim=0)

        return y, a_ij    