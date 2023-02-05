import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from CC.crf import CRF


class BiRnnCrf(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_rnn_layers=1, rnn="lstm"):
        super(BiRnnCrf, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        self.fc1 = nn.Linear(embedding_dim + hidden_dim, self.tagset_size + 2)
        self.dense = nn.Linear(embedding_dim * 4, embedding_dim)
        self.act = nn.GELU()
        self.layerNorm = nn.LayerNorm(embedding_dim)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(True),
            nn.Linear(embedding_dim, embedding_dim)
        )

        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(self.embedding_dim, hidden_dim // 2, num_layers=num_rnn_layers,
                       bidirectional=True, batch_first=True)
        self.crf = CRF(hidden_dim, self.tagset_size)

    def __build_features(self, embeds, masks):

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length.cpu(), batch_first=True)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks

    def loss(self, embeds, prompt, masks, tags):
        features, masks = self.__build_features(embeds, masks)
        prompt_feature = self.dense(prompt)
        
        prompt_feature_weights = self.attention(prompt_feature)
        prompt_feature_weights = torch.softmax(prompt_feature_weights, dim=1)
        prompt_feature = prompt_feature * prompt_feature_weights
        
        prompt_feature = self.act(prompt_feature)
        prompt_feature = self.layerNorm(prompt_feature)
        fusion = torch.cat([features, prompt_feature[:, :features.shape[1], :]], dim=-1)
        fusion = self.fc1(fusion)
        loss = self.crf.loss(fusion, tags, masks=masks)
        return loss

    def forward(self, embeds, prompt, masks):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(embeds, masks)
        prompt_feature = self.dense(prompt)
        
        prompt_feature_weights = self.attention(prompt_feature)
        prompt_feature_weights = torch.softmax(prompt_feature_weights, dim=1)
        prompt_feature = prompt_feature * prompt_feature_weights
        
        prompt_feature = self.act(prompt_feature)
        prompt_feature = self.layerNorm(prompt_feature)
        fusion = torch.cat([features, prompt_feature[:, :features.shape[1], :]], dim=-1)
        fusion = self.fc1(fusion)
        scores, tag_seq = self.crf(fusion, masks)
        return scores, tag_seq