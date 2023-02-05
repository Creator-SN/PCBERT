"""
implement of LEBERT
"""

import math
import os
import warnings

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import BertConfig
from CC.crf import CRF

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPast
from transformers import load_tf_weights_in_bert
BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """
    Construct the embeddingns fron word, position and token_type, boundary embeddings
    """

    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(
            config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertLayer(nn.Module):
    """
    we modify the module to add word embedding information into the transformer
    """

    def __init__(self, config, has_word_attn=False):
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)

        # here we add a attention for matched word
        self.has_word_attn = has_word_attn
        if self.has_word_attn:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.act = nn.Tanh()

            self.word_transform = nn.Linear(
                config.word_embed_dim, config.hidden_size)
            self.word_word_weight = nn.Linear(
                config.hidden_size, config.hidden_size)
            attn_W = torch.zeros(config.hidden_size, config.hidden_size)
            self.attn_W = nn.Parameter(attn_W)
            self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)
            self.fuse_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps)
            
            self.word_label_transform = nn.Linear(config.word_embed_dim + config.label_embed_dim, config.word_embed_dim)
            self.label_label_weight = nn.Linear(config.word_embed_dim, config.word_embed_dim)
            attn_Label_W = torch.zeros(config.word_embed_dim, config.word_embed_dim)
            self.attn_Label_W = nn.Parameter(attn_Label_W)
            self.attn_Label_W.data.normal_(mean=0.0, std=config.initializer_range)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            input_word_embeddings=None,
            input_label_embeddings=None,
            input_word_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False
    ):
        """
        code refer to: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py
        B: batch_size
        L: seq length
        W: word size
        D: word_embedding dim
        Args:
            input_word_embedding: [B, L, W, D]
            input_word_mask: [B, L, W]
        """
        # 1.character contextual representation
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # this is the contextual representation
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        # decode need join attention from the outputs
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        if self.has_word_attn:
            assert input_word_mask is not None

            # h_ori = torch.repeat_interleave(layer_output.unsqueeze(2), repeats=input_label_embeddings.shape[2], dim=2)

            # input_word_embeddings [Batch_size, Seq_len, Word_size, Label_size, Dim]
            label_attn_score = torch.matmul(input_word_embeddings.unsqueeze(3), self.attn_Label_W) # [Batch_size, Seq_len, Word_size, 1, Dim]
            label_attn_score = torch.matmul(label_attn_score, torch.transpose(input_label_embeddings, 3, 4)) # [Batch_size, Seq_len, Word_size, 1, Label_size]
            label_attn_score = label_attn_score.squeeze() # [Batch_size, Seq_len, Word_size, Label_size]
            label_attn_score = torch.nn.Softmax(dim=-1)(label_attn_score)
            label_attn_score = label_attn_score.unsqueeze(-1) # [Batch_size, Seq_len, Word_size, Label_size, 1]
            
            sum_label_embeddings = torch.sum(input_label_embeddings * label_attn_score, dim=3)
            label_feature = self.word_label_transform(torch.cat([input_word_embeddings, sum_label_embeddings], dim=-1))
            label_feature = self.act(label_feature)
            label_feature = self.label_label_weight(label_feature)
            label_feature = self.dropout(label_feature)
            input_word_fusion = input_word_embeddings + label_feature
            # transform word_dim + label_dim => 768 
            word_outputs = self.word_transform(
                input_word_fusion)  # [Batch_size, Seq_L, Words_sum, Dim]
            word_outputs = self.act(word_outputs)
            word_outputs = self.word_word_weight(word_outputs)
            word_outputs = self.dropout(word_outputs)

            # attention_output = attention_output.unsqueeze(2) # [Batch_size, Seq_L, Dim] -> [Batch_size, Seq_L, 1, Dim]
            alpha = torch.matmul(layer_output.unsqueeze(2),
                                 self.attn_W)  # [Batch_size, Seq_L, 1, Dim]
            alpha = torch.matmul(alpha, torch.transpose(
                word_outputs, 2, 3))  # [Batch_size, Seq_L, 1, Words_sum]
            alpha = alpha.squeeze()  # [Batch_size, Seq_L, Words_sum]
            alpha = alpha + (1 - input_word_mask.float()) * (-10000.0)
            alpha = torch.nn.Softmax(dim=-1)(alpha)  # [Batch_size, Seq_L, Words_sum]
            alpha = alpha.unsqueeze(-1)  # [Batch_size, Seq_L, Words_sum, 1]
            weighted_word_embedding = torch.sum(
                word_outputs * alpha, dim=2)  # [Batch_size, Seq_L, Dim]
            layer_output = layer_output + weighted_word_embedding

            layer_output = self.dropout(layer_output)
            layer_output = self.fuse_layernorm(layer_output)

        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_layers = config.add_layers

        total_layers = []
        for i in range(config.num_hidden_layers):
            if i in self.add_layers:
                total_layers.append(BertLayer(config, True))
            else:
                total_layers.append(BertLayer(config, False))

        self.layer = nn.ModuleList(total_layers)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            input_word_embeddings=None,
            input_label_embeddings=None,
            input_word_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # print("Layer 0: \n")
        # print(hidden_states)
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    input_word_embeddings,
                    input_label_embeddings,
                    input_word_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    input_word_embeddings,
                    input_label_embeddings,
                    input_word_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            # print("Layer %d: \n"%(i+1))
            # print(hidden_states)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class PCBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super(PCBertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        matched_word_embeddings=None,
        matched_label_embeddings=None,
        matched_word_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

        batch_size: N
        seq_length: L
        dim: D
        word_num: W


        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            matched_word_embeddings: [B, L, W, D]
            matched_label_embeddings: [B, L, W, D]
            matched_word_mask: [B, L, W]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            input_word_embeddings=matched_word_embeddings,
            input_label_embeddings=matched_label_embeddings,
            input_word_mask=matched_word_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
