import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel
from CC.PCBert import PCBertModel
from CC.LEBert import WCBertModel
from CC.birnncrf import BiRnnCrf
from ICCSupervised.ICCSupervised import IModel


class CCNERModel(IModel):

    def __init__(self, **args):
        required_pretrained_embedding_models = [
            'LEBert', 'LEBertFusion', 'PLEBert']
        required_label_embedding_models = ['PLEBert']
        assert "model_name" in args, "argument model_name required"
        assert "bert_config_file_name" in args, "argument bert_config_file_name required"
        assert "pretrained_file_name" in args, "argument pretrained_file_name required"
        assert "tagset_size" in args, "argument tagset_size required"
        assert "hidden_dim" in args, "argument hidden_dim required"
        self.model_name = args['model_name']
        self.bert_config_file_name = args['bert_config_file_name']
        self.pretrained_file_name = args['pretrained_file_name']
        self.tagset_size = args['tagset_size']
        self.hidden_dim = args['hidden_dim']
        if args['model_name'] in required_pretrained_embedding_models:
            assert "pretrained_embeddings" in args, "argument pretrained_embeddings required"
            self.pretrained_embeddings = args['pretrained_embeddings']
        if args['model_name'] in required_label_embedding_models:
            assert "label_embeddings" in args, "argument label_embeddings required"
            self.label_embeddings = args['label_embeddings']
        self.load_model()

    def load_model(self, LSTM=True):
        config = BertConfig.from_json_file(self.bert_config_file_name)
        if self.model_name == 'PLEBert':
            self.model = PLEBertModel.from_pretrained(
                self.pretrained_file_name, pretrained_embeddings=self.pretrained_embeddings, label_embeddings=self.label_embeddings, config=config)
        if self.model_name == 'LEBert':
            self.model = LEBertModel.from_pretrained(
                self.pretrained_file_name, pretrained_embeddings=self.pretrained_embeddings, config=config)
        self.birnncrf = BiRnnCrf(
            tagset_size=self.tagset_size, embedding_dim=config.hidden_size, hidden_dim=self.hidden_dim)

    def get_model(self):
        return self.model, self.birnncrf

    def __call__(self):
        return self.get_model()


class BertBaseModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, **args):
        input = {
            'input_ids': args['input_ids'],
            'attention_mask': args['attention_mask']
        }
        if 'token_type_ids' in args:
            input['token_type_ids'] = args['token_type_ids']
        outputs = self.bert(**input)
        return {
            'mix_output': outputs.last_hidden_state,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

class PLEBertModel(BertPreTrainedModel):
    '''
    config: BertConfig
    pretrained_embeddings: 预训练embeddings shape: size * 200
    '''

    def __init__(self, config, pretrained_embeddings, label_embeddings):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)

        label_vocab_size = label_embeddings.shape[0]
        label_dim = label_embeddings.shape[1]
        self.label_embeddings = nn.Embedding(label_vocab_size, label_dim)
        self.bert = PCBertModel(config)

        self.init_weights()

        # init the embedding
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        self.label_embeddings.weight.data.copy_(
            torch.from_numpy(label_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            **args
    ):
        matched_word_embeddings = self.word_embeddings(
            args['matched_word_ids'])
        matched_label_embeddings = self.label_embeddings(
            args['matched_label_ids'])
        outputs = self.bert(
            input_ids=args['input_ids'],
            attention_mask=args['attention_mask'],
            token_type_ids=args['token_type_ids'],
            matched_word_embeddings=matched_word_embeddings,
            matched_label_embeddings=matched_label_embeddings,
            matched_word_mask=args['matched_word_mask']
        )

        sequence_output = outputs[0]

        return {
            'mix_output': sequence_output,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

class LEBertModel(BertPreTrainedModel):
    '''
    config: BertConfig
    pretrained_embeddings: 预训练embeddings shape: size * 200
    '''

    def __init__(self, config, pretrained_embeddings):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        self.bert = WCBertModel(config)

        self.init_weights()

        # init the embedding
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            **args
    ):
        matched_word_embeddings = self.word_embeddings(
            args['matched_word_ids'])
        if 'prompt_features' not in args:
            args['prompt_features'] = None
        outputs = self.bert(
            input_ids=args['input_ids'],
            attention_mask=args['attention_mask'],
            token_type_ids=args['token_type_ids'],
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=args['matched_word_mask'],
            prompt_features=args['prompt_features']
        )

        sequence_output = outputs[0]

        return {
            'mix_output': sequence_output,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
