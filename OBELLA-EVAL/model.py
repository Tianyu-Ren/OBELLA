import torch
from transformers import AlbertModel, AlbertConfig, AutoTokenizer
from torch import nn
import math


class OBELLA(nn.Module):

    def __init__(self, attention_func, pooling, num_class=3, max_length=256, post_trained_weight=None, device='cuda:0'):
        super().__init__()

        self.encoder = AlbertModel(AlbertConfig.from_pretrained("albert/albert-xxlarge-v2"))

        args = self.encoder.config

        self.cross_attention = CrossAttentionModule(args.hidden_size, attention_func, args.num_attention_heads,
                                                    args.attention_probs_dropout_prob)

        self.feedforward = FeedForwardModule(args.hidden_size, args.intermediate_size)

        self.add_norm1 = AddNorm(args.hidden_size, args.hidden_dropout_prob)

        self.add_norm2 = AddNorm(args.hidden_size, args.hidden_dropout_prob)

        self.pooling = pooling

        self.head = nn.Linear(args.hidden_size, num_class)

        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained("albert/albert-xxlarge-v2")

        self.max_length = max_length

        if post_trained_weight is not None:
            state_dict = torch.load(post_trained_weight)
            state_dict = {k.replace('module.', ''):v for k,v in state_dict.items()}
            self.load_state_dict(state_dict)
        else:
            self.encoder = AlbertModel.from_pretrained("albert/albert-xxlarge-v2")
        self.device = device
        self.to(device)

    def forward(self, question: (list, tuple), reference, candidate):
        inputs, reference_span, candidate_span = get_tensor_inputs(self.tokenizer, question, reference,
                                                                   candidate, self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        x = self.encoder(**inputs).last_hidden_state
        # select span
        # x_reference -> (batch_size, n, model_dim); attention_mask -> (batch_size, n)
        x_reference, reference_mask = get_span_representation(x, reference_span)
        reference_mask = torch.repeat_interleave(reference_mask, self.args.num_attention_heads, 0)
        # x_candidate -> (batch_size, t, model_dim)
        x_candidate, candidate_mask = get_span_representation(x, candidate_span)
        # Pass to cross attention
        x_candidate = self.add_norm1(x_candidate, self.cross_attention(x_candidate, x_reference, reference_mask))
        # Projection
        x_candidate = self.add_norm2(x_candidate, self.feedforward(x_candidate))
        # Pooling
        x_candidate = self.pooling(x_candidate, candidate_mask)
        # Classification
        return self.head(x_candidate)


class CrossAttentionModule(nn.Module):
    def __init__(self, model_dim, attention_func, num_heads, dropout):
        super().__init__()
        self.Q = nn.Linear(model_dim, model_dim, bias=False)
        self.K = nn.Linear(model_dim, model_dim, bias=False)
        self.V = nn.Linear(model_dim, model_dim, bias=False)
        self.O = nn.Linear(model_dim, model_dim, bias=False)
        self.attention = attention_func(dropout)
        self.num_heads = num_heads

    def forward(self, query, key, attention_mask):
        # Projection
        Q, K, V = self.Q(query), self.K(key), self.V(key)
        # Split Projection
        Q, K, V = head_split(Q, self.num_heads), head_split(K, self.num_heads), head_split(V, self.num_heads)
        # Modeling dependencies
        V = head_concat(self.attention(Q, K, V, attention_mask), self.num_heads)
        return self.O(V)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask):
        # input_size: batch_size, sequence_length, model_dimension
        # Normalization Factor
        model_dim = query.shape[2]
        attention_score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(model_dim)
        attention_weight = masked_softmax(attention_score, attention_mask)
        return torch.bmm(self.dropout(attention_weight), value)


def get_span_representation(x, span_indicator):
    """
    :param x: x here is the encoded complete batch of sequence (batch_size, sequence_length, model_dim)
    :param span_indicator: indicates where the reference/candidate in the span
    :return:
    """
    batch_size = x.shape[0]
    span_length = span_indicator[:, 1] - span_indicator[:, 0]
    max_length = span_length.max().item()
    attention_mask = torch.tensor([[1] * i + [0] * (max_length - i) for i in span_length], device=x.device)
    spans = []
    for n in range(batch_size):
        span = x[n, span_indicator[n][0]: span_indicator[n][1], :]

        if max_length > len(span):
            span = torch.concat([span, torch.zeros(max_length - len(span), x.shape[-1], device=x.device)])

        spans.append(span)
    return torch.stack(spans, 0), attention_mask


def masked_softmax(x, attention_mask):
    """
    :param x: The x here is expected to be the attention score in the shape of (batch_size, query_length, key_length)
    :param attention_mask: (batch_size, key_length)
    :return:
    """
    attention_mask = torch.repeat_interleave(torch.unsqueeze(attention_mask, 1), x.shape[1], 1)
    x[attention_mask == 0] += -1e6
    return nn.functional.softmax(x, dim=-1)


def masked_pooling(x, attention_mask):
    attention_mask_ = torch.repeat_interleave(torch.unsqueeze(attention_mask, -1), x.shape[-1], 2)
    x[attention_mask_ == 0] = 0
    return x.sum(dim=1) / attention_mask.sum(dim=1).reshape(-1, 1)


def head_split(x, num_heads):
    """
    :param x: input tensor: Q, K or V
    :param num_heads: number of attention heads
    :return: split x
    """
    # Split
    shape = x.shape
    x = x.reshape(shape[0], shape[1], num_heads, -1)
    # After split
    # x (batch_size, sequence_length, model_dim) ----> (batch_size, sequence_length, num_heads, model_dim // num_heads)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, x.shape[2], x.shape[3])


def head_concat(x, num_heads):
    """
    :param x: split input tensor: Q, K or V (batch_size * num_heads, sequence_length, model_dim // num_heads)
    :param num_heads: number of attention heads
    :return: split x
    """
    shape = x.shape
    x = x.reshape(-1, num_heads, shape[1], shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


class FeedForwardModule(nn.Module):
    def __init__(self, model_dim, intermediate_dim):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(intermediate_dim, model_dim)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.layer_norm(self.dropout(y) + x)


def get_tensor_inputs(tokenizer, question: list, reference: list, candidate: list, max_length):

    def text_concat(text: (list, tuple), concat_token):
        concat_text_ = text[0]  # Expected to be questions list
        for i in text[1:]:
            concat_text_ = list(map(lambda x, y: x + concat_token + concat_token + y, concat_text_, i))
        return concat_text_

    sep_token, sep_token_id = tokenizer.sep_token, tokenizer.sep_token_id
    concat_text = text_concat([question, reference, candidate], sep_token)
    tensors = tokenizer.batch_encode_plus(concat_text, max_length=max_length,
                                          padding="max_length", truncation=True, return_tensors="pt")
    indicators = torch.repeat_interleave(torch.unsqueeze(torch.arange(max_length), 0), len(question), 0)

    sep_indices = indicators[tensors.input_ids == sep_token_id].reshape(-1, 5)

    reference_span = torch.stack([sep_indices[:, 1] + 1, sep_indices[:, 2]], dim=0).T

    candidate_span = torch.stack([sep_indices[:, 3] + 1, sep_indices[:, 4]], dim=0).T

    return tensors, reference_span, candidate_span
