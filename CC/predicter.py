import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertTokenizer, BertModel
from ICCSupervised.ICCSupervised import IPredict
from CC.loaders import *
from CC.loaders.utils import *
from CC.model import *
from CC.dataloader import AutoDataLoader
from CC.analysis import CCAnalysis


class NERPredict(IPredict):

    def __init__(self, **args):
        KwargsParser(debug=True) \
            .add_argument("use_gpu", bool, defaultValue=False) \
            .add_argument("loader_name", str, defaultValue="le_loader") \
            .add_argument("model_name", str, defaultValue="LEBert") \
            .add_argument("lstm_crf_model_file", str) \
            .add_argument("bert_model_file", str) \
            .add_argument("hidden_dim", int) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("bert_config_file_name", str) \
            .add_argument("tag_file", str) \
            .add_argument("padding_length", int, 512) \
            .add_argument("num_gpus", list, [0]) \
            .parse(self, **args)
        args["use_test"] = True
        args["do_predict"] = True
        self.dataloader_init(**args)
        self.model_init()

    def dataloader_init(self, **args):
        self.dataloader = AutoDataLoader(**args)
        result = self.dataloader()
        if self.loader_name == 'le_loader':
            self.vocab_embedding = result['vocab_embedding']
            self.embedding_dim = result['embedding_dim']
            self.tag_vocab = result['tag_vocab']
            self.tag_size = self.tag_vocab.__len__()
            self.analysis = CCAnalysis(
                self.tag_vocab.token2id, self.tag_vocab.id2token)
        if self.loader_name == 'cn_loader':
            self.dm = result['dm']
            self.tag_size = len(self.dm.tag_to_idx)
            self.analysis = CCAnalysis(self.dm.tagToIdx, self.dm.idxToTag)

    def model_init(self):
        config = BertConfig.from_json_file(self.bert_config_file_name)
        if self.model_name == 'LEBert':
            self.model = LEBertModel(
                config, pretrained_embeddings=self.vocab_embedding)
        elif self.model_name == 'LEBertFusion':
            self.model = LEBertModelFusion(
                config, pretrained_embeddings=self.vocab_embedding)
        elif self.model_name == 'Bert':
            self.model = BertBaseModel(config)

        if torch.cuda.is_available() and self.use_gpu:
            bert_dict = torch.load(self.bert_model_file)
            if isinstance(bert_dict,nn.DataParallel):
                bert_dict = bert_dict.module.state_dict()
            self.model.load_state_dict(bert_dict)
            self.birnncrf = torch.load(self.lstm_crf_model_file).cuda()
            if torch.cuda.device_count() > 1 and len(self.num_gpus)>0:
                self.model = nn.DataParallel(
                    self.model, device_ids=self.num_gpus)
                self.birnncrf = self.birnncrf.cuda()
        else:
            bert_dict = torch.load(
                self.bert_model_file, map_location="cpu")
            if isinstance(bert_dict,nn.DataParallel):
                bert_dict = bert_dict.module.state_dict()
            self.model.load_state_dict(bert_dict)
            self.birnncrf = torch.load(
                self.lstm_crf_model_file, map_location="cpu")

        self.model.eval()
        self.birnncrf.eval()

    # def data_process(self, sentences):
    #     result = []
    #     pad_tag = '[PAD]'
    #     if type(sentences) == str:
    #         sentences = [sentences]
    #     max_len = 0
    #     for sentence in sentences:
    #         encode = self.tokenizer.encode(
    #             sentence, add_special_tokens=True, max_length=self.padding_length, truncation=True)
    #         result.append(encode)
    #         if max_len < len(encode):
    #             max_len = len(encode)

    #     for i, sentence in enumerate(result):
    #         remain = max_len - len(sentence)
    #         for _ in range(remain):
    #             result[i].append(self.dm.wordToIdx(pad_tag))
    #     return torch.tensor(result)

    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available() and self.use_gpu:
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available() and self.use_gpu:
                return inputX.cuda()
            return inputX

    def pred(self, sentences, return_dict: bool = False):
        # sentences = self.data_process(sentences)
        device = None
        if self.use_gpu:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        self.model.to(device)
        self.birnncrf.to(device)

        if self.loader_name == "le_loader":
            batch = {}
            new_sentence = []
            for sentence in sentences:
                new_sentence.append(list(sentence))
                it = self.dataloader.loader.myData_test.convert_embedding(
                    {"text": new_sentence[-1]}, to_tensor=False, return_dict=True)
                for key in it.keys():
                    if key not in batch:
                        batch[key] = []
                    batch[key].append(it[key])
            it = batch
            with torch.no_grad():
                for key in it.keys():
                    it[key] = self.cuda(torch.tensor(it[key]))
                outputs = self.model(**it)
                hidden_states = outputs['mix_output']
                pred = self.birnncrf(
                    hidden_states, it['input_ids'].gt(0))[1]
                pred_tags = [self.analysis.idx2tag(it)[1:-1] for it in pred]
                if not return_dict:
                    return list(zip(new_sentence, pred_tags))
                else:
                    return [dict(zip(sentence, tags)) for sentence, tags in zip(new_sentence, pred_tags)]
        else:
            # TODO: implement cn_loader
            raise NotImplementedError()

    def __call__(self, sentences, return_dict: bool = False):
        return self.pred(sentences, return_dict=return_dict)
