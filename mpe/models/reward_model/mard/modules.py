import torch
from torch import nn
import torch.nn.functional as F
import math

""" encoder layer """
class EncoderLayer(nn.Module):
    def __init__(self, d_hidden, n_head, d_ff, dropout, layer_norm_epsilon=1e-12):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.emb_dropout = dropout
        self.d_ff = d_ff
        self.layer_norm_epsilon = layer_norm_epsilon
        
        self.self_attn = MultiHeadAttention(self.d_hidden, self.n_head, self.emb_dropout)
        self.layer_norm1 = nn.LayerNorm(self.d_hidden, eps=self.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.d_hidden, self.d_ff, self.emb_dropout)
        self.layer_norm2 = nn.LayerNorm(self.d_hidden, eps=self.layer_norm_epsilon)
    
    def forward(self, inputs, attn_mask):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        # att_inputs = self.layer_norm1(inputs)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        # att_outputs = inputs + att_outputs
        ffn_inputs = self.layer_norm1(inputs + att_outputs)
        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(ffn_inputs) + att_outputs
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob

""" multi agent attention """
class MultiHeadAttention(nn.Module):
    def __init__(self, d_hidden, n_head, dropout):
        super().__init__()
        if d_hidden%n_head != 0:
            raise ValueError(
                f"The hidden size ({d_hidden}) is not a multiple of the number of attention "
                f"heads ({n_head})"
            )
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.emb_dropout = dropout
        self.attn_head_size = int(self.d_hidden/self.n_head)
        self.all_head_size = self.attn_head_size*self.n_head

        self.W_Q = nn.Linear(self.d_hidden, self.all_head_size)
        self.W_K = nn.Linear(self.d_hidden, self.all_head_size)
        self.W_V = nn.Linear(self.d_hidden, self.all_head_size)
        self.scaled_dot_attn = ScaledDotProductAttention(self.emb_dropout, self.attn_head_size)
        self.linear = nn.Linear(self.all_head_size, self.d_hidden)
        self.dropout = nn.Dropout(self.emb_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_head, self.attn_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V, attn_mask):

        # (bs, n_head, n_q_seq, attn_head_size)
        q_s = self.transpose_for_scores(self.W_Q(Q))
        # (bs, n_head, n_k_seq, attn_head_size)
        k_s = self.transpose_for_scores(self.W_K(K))
        # (bs, n_head, n_v_seq, attn_head_size)
        v_s = self.transpose_for_scores(self.W_V(V))

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (bs, n_head, n_q_seq, all_head_size)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)
        
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob

class MultiAgentAttention(nn.Module):
    def __init__(self, d_hidden, n_head, n_agents, dropout, device):
        super().__init__()
        if d_hidden%n_head != 0:
            raise ValueError(
                f"The hidden size ({d_hidden}) is not a multiple of the number of attention "
                f"heads ({n_head})"
            )
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.n_agents = n_agents
        self.emb_dropout = dropout
        self.device = device
        self.attn_head_size = int(self.d_hidden/self.n_head)
        self.all_head_size = self.attn_head_size*self.n_head

        self.W_Q = nn.ModuleList([nn.Linear(self.d_hidden, self.all_head_size) for _ in range(self.n_agents)])
        self.W_K = nn.Linear(self.d_hidden, self.all_head_size)
        self.W_V = nn.Linear(self.d_hidden, self.all_head_size)
        self.scaled_dot_attn = ScaledDotProductAttention(self.emb_dropout, self.attn_head_size)
        self.linear = nn.Linear(self.all_head_size, self.d_hidden)
        self.dropout = nn.Dropout(self.emb_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_head, self.attn_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V, attn_mask):

        # (bs, n_head, n_q_seq, attn_head_size)
        Q = torch.cat([self.W_Q[i](Q[:,i,:]).unsqueeze(dim=1) for i in range(self.n_agents)], dim=1)
        q_s = self.transpose_for_scores(Q)
        # q_s = self.transpose_for_scores(self.W_Q(Q))
        # (bs, n_head, n_k_seq, attn_head_size)
        k_s = self.transpose_for_scores(self.W_K(K))
        # (bs, n_head, n_v_seq, attn_head_size)
        v_s = self.transpose_for_scores(self.W_V(V))

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (bs, n_head, n_q_seq, all_head_size)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)
        
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob
    
""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout, attn_head_size):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attn_head_size = attn_head_size
    
    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores / math.sqrt(self.attn_head_size)

        scores.masked_fill_(attn_mask==0, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)
        
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob


""" feed forward """
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_hidden, d_ff, dropout):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_ff = d_ff

        self.dense1 = nn.Linear(d_hidden, d_ff)
        self.dense2 = nn.Linear(d_ff, d_hidden)
        # self.conv1 = nn.Conv1d(in_channels=self.d_hidden, out_channels=self.d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=self.d_hidden, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):

        # (bs, n_seq, d_ff)
        output = self.dense1(inputs)
        output = self.active(output)
        # (bs, n_seq, d_hidn)
        output = self.dense2(output)
        output = self.dropout(output)

        return output