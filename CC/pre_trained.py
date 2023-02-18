# %%
import os
import re
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
from tqdm import tqdm
from ICCSupervised.ICCSupervised import ITrainer
from CC.dataloader import AutoDataLoader
from CC.analysis import CCAnalysis
from CC.model import CCNERModel
from CC.loaders.utils import *


class NERPreTrainer(ITrainer):

    def __init__(self, **args):
        assert "num_epochs" in args, "argument num_epochs: required embeding file path"
        assert "num_gpus" in args, "argument num_gpus: required embeding file path"
        self.loader_name = 'lex_loader'
        if "loader_name" in args:
            self.loader_name = args["loader_name"]
        self.model_name = 'Bert'
        if "model_name" in args:
            self.model_name = args["model_name"]

        self.num_epochs = args['num_epochs']
        self.num_gpus = args['num_gpus']
        self.dataloader_init(**args)
        self.model_init(**args)
        self.task_name = args['task_name']

    def model_init(self, **args):
        config = BertConfig.from_json_file(args['bert_config_file_name'])
        self.model = BertForMaskedLM.from_pretrained(
            args['pretrained_file_name'], config=config)

    def dataloader_init(self, **args):
        self.dataloader = AutoDataLoader(**args)
        result = self.dataloader()
        self.train_data = result['train_set']
        self.train_iter = result['train_iter']
        if self.loader_name in ['lex_loader', 'labellex_loader']:
            self.vocab_embedding = result['vocab_embedding']
            self.embedding_dim = result['embedding_dim']
            self.tag_vocab = result['tag_vocab']
            self.tag_size = self.tag_vocab.__len__()
            self.analysis = CCAnalysis(
                self.tag_vocab.token2id, self.tag_vocab.id2token)
        elif self.loader_name == 'cnx_loader':
            self.dm = result['dm']
            self.tag_size = len(self.dm.tag_to_idx)
            self.analysis = CCAnalysis(self.dm.tagToIdx, self.dm.idxToTag)
        elif self.loader_name == 'ptloader_v2':
            self.analysis = CCAnalysis(None, None)

    def train(self, lr=1e-4):

        optimizer = optim.AdamW(self.model.parameters(),
                                lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 0:
            self.model = nn.DataParallel(self.model, device_ids=self.num_gpus)

        self.model.to(device)

        train_step = 0
        train_steps = []
        epoch_train_loss = []
        for epoch in range(self.num_epochs):
            train_count = 0
            train_loss = []
            train_iter = tqdm(self.train_iter)
            self.model.train()
            for it in train_iter:
                train_step += 1

                for key in it.keys():
                    it[key] = self.cuda(it[key].long())

                outputs = self.model(input_ids=it['input_ids'], attention_mask=it['attention_mask'],
                                     token_type_ids=it['token_type_ids'], labels=it['origin_labels'])
                loss = outputs.loss
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()

                train_loss.append(loss.data.item())
                train_count += 1

                train_iter.set_description(
                    'Epoch: {}/{} Train'.format(epoch + 1, self.num_epochs))
                train_iter.set_postfix(train_loss=np.mean(train_loss))
                self.analysis.append_train_record({
                    'loss': loss.data.item()
                })

            model_uid = self.save_model(train_step)
            train_steps.append(train_step)
            epoch_train_loss.append(np.mean(train_loss))

            self.analysis.save_csv('./data_record/{}'.format(self.task_name),
                                   '{}_loss'.format(self.model_name), loss=epoch_train_loss, step=train_steps)
            yield (epoch + 1, self.analysis.train_record, model_uid)

    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists('./save_pretrained/{}'.format(dir)):
            os.makedirs('./save_pretrained/{}'.format(dir))
        if torch.cuda.device_count() > 0:
            self.model.module.save_pretrained(
                './save_pretrained/{}/{}_{}'.format(dir, self.model_name, current_step))
        else:
            self.model.save_pretrained(
                './save_pretrained/{}/{}_{}'.format(dir, self.model_name, current_step))
        self.analysis.append_model_record(current_step)
        return current_step

    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX

    def __call__(self, **arg):
        for r in self.train(**arg):
            yield r
