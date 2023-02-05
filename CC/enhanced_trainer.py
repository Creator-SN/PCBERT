import json
import os
import re
import uuid
import torch
import numpy as np
from torch.autograd.grad_mode import F
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertTokenizer, BertModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from ICCSupervised.ICCSupervised import ITrainer
from CC.dataloader import AutoDataLoader
from CC.analysis import CCAnalysis
from CC.model import CCNERModel
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score,classification_report


class EnhancedNERTrainer():

    def __init__(self, **args):
        '''
        Args:
        - num_epochs
        - num_gpus
        - bert_config_file_name
        - pretrained_file_name
        - prompt_pretrained_file_name
        - hidden_dim
        - word_embedding_file: required in `le_loader`
        - word_vocab_file: required in `le_loader`
        - train_file
        - eval_file
        - test_file
        - tag_file
        - bert_vocab_file
        - batch_size
        - eval_batch_size
        - word_tag_split: optional in `cn_loader`, default: ' '
        - pattern: optional in `cn_loader`, default: '， O'
        - output_eval: optional, default: None
        - max_scan: optional in `le_loader`, default: None
        - add_seq_vocab: optional in `le_loader`, default: False
        - max_seq_length: optional in `le_loader`, default: 256
        - max_word_num: optional in `le_loader`, default: 5
        - default_tag: optional in `le_loader`, default: "O"
        - model_name: optional, default: "LEBert"
        - loader_name: optional, default: "le_loader"
        - task_name: optional, default: None
        '''
        assert "num_epochs" in args, "argument num_epochs: required embeding file path"
        assert "num_gpus" in args, "argument num_gpus: required embeding file path"
        assert "hidden_dim" in args, "argument hidden_dim: required embeding file path"
        self.model_name: str = 'LEBert'
        if "model_name" in args:
            self.model_name = args["model_name"]
        self.loader_name = 'le_loader'
        if "loader_name" in args:
            self.loader_name = args["loader_name"]

        self.eval_data = args['output_eval']
        self.num_epochs = args['num_epochs']
        self.num_gpus = args['num_gpus']
        self.output_eval = args['output_eval']
        self.hidden_dim = args['hidden_dim']
        self.max_seq_length = args['max_seq_length']
        self.dataloader_init(**args)
        self.model_init(**args)
        self.task_name = args['task_name']

    def model_init(self, **args):
        model_args = {
            'model_name': self.model_name,
            'bert_config_file_name': args['bert_config_file_name'],
            'tagset_size': self.tag_size,
            'hidden_dim': args['hidden_dim'],
        }
        if 'word_embedding_file' in args:
            model_args['pretrained_embeddings'] = self.vocab_embedding
        if 'tag_embedding_file' in args:
            model_args['label_embeddings'] = self.label_embedding
        if 'pretrained_file_name' in args:
            model_args['pretrained_file_name'] = args['pretrained_file_name']

        self.bert_ner = CCNERModel(**model_args)
        self.model, self.birnncrf = self.bert_ner()
        self.prompt_model = BertModel.from_pretrained(
            args['prompt_pretrained_file_name'], config=args['prompt_config_file_name'])

    def dataloader_init(self, **args):
        self.dataloader = AutoDataLoader(**args)
        result = self.dataloader()
        self.train_data = result['train_set']
        self.train_iter = result['train_iter']
        
        if self.loader_name == 'ft_loader_v4':
            self.vocab_embedding = result['vocab_embedding']
            self.embedding_dim = result['embedding_dim']
            self.tag_vocab = result['tag_vocab']
            self.tag_size = self.tag_vocab.__len__()
            self.analysis = CCAnalysis(
                self.tag_vocab.token2id, self.tag_vocab.id2token)

        if self.output_eval is not None:
            self.eval_set = result['eval_set']
            self.eval_iter = result['eval_iter']

    def train(self, resume_path=False, resume_step=False, lr1=2e-5, lr2=1e-3, eval_call_epoch=None, train = True,eval = True, save = True):
        
        optimizer = optim.AdamW([
            {'params': self.model.parameters(), 'lr': lr1},
            {'params': self.birnncrf.parameters(), 'lr': lr2}
        ], lr=1e-5, weight_decay=0.)
        
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 0:
            self.model = nn.DataParallel(self.model, device_ids=self.num_gpus)
            self.prompt_model = nn.DataParallel(self.prompt_model, device_ids=self.num_gpus)
            self.birnncrf.cuda()

        self.model.to(device)
        self.birnncrf.to(device)

        self.prompt_model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} step: {} ...\n'.format(
                resume_path, resume_step))
            bert_model_dict = torch.load(os.path.join(
                resume_path, self.model_name, '{}_{}.pth'.format(self.model_name, resume_step))).module.state_dict()
            self.model.module.load_state_dict(bert_model_dict)
            self.birnncrf = torch.load(os.path.join(
                resume_path, 'lstm_crf', 'lstm_crf_{}.pth'.format(resume_step)))
            self.model.to(device)
            self.birnncrf.to(device)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step != False else 0
        
        for epoch in range(self.num_epochs):
            if train:
                train_count = 0
                train_loss = 0
                train_iter = tqdm(self.train_iter)
                self.model.train()
                self.birnncrf.train()
                self.prompt_model.eval()

                all_acc_list = []
                all_p_list = []
                all_r_list = []
                all_f1_list = []
                all_preds = []
                all_trues = []
                for it in train_iter:
                    pred_labels_list = []
                    true_labels_list = []

                    train_step += 1

                    for key in it.keys():
                        it[key] = self.cuda(it[key])
                    
                    # [batch_size, 4, 512]
                    prompt_inputs = it['prompt_input_ids']
                    prompt_inputs = prompt_inputs.reshape(-1, 512) # [batch_size * 7, 512]
                    prompt_origin_labels = it['prompt_origin_labels']
                    prompt_origin_labels = prompt_origin_labels.reshape(-1, 512) # [batch_size * 7, 512]
                    prompt_attention_mask = prompt_origin_labels.gt(0)
                    prompt_indexed = []
                    for i in range(it['input_ids'].shape[0]):
                        it['prompt_indexes'][i] = it['prompt_indexes'][i] + 2048 * i
                        prompt_indexed += it['prompt_indexes'][i]
                    
                    prompt_outputs = self.prompt_model(input_ids=prompt_inputs, attention_mask=prompt_attention_mask)
                    prompt_hidden_states = prompt_outputs.last_hidden_state # [batch_size * 7, 512, 768]
                    prompt_hidden_states = prompt_hidden_states * prompt_attention_mask.unsqueeze(-1).float() # [batch_size * 7, 512, 768]
                    prompt_hidden_states = prompt_hidden_states.reshape(-1, 768) # [batch_size * 7 * 512, 768]
                    prompt_entity_hs = prompt_hidden_states[prompt_indexed] # [batch_size * max_seq_len * entity_pad_len(4), 768]
                    prompt_entity_hs = prompt_entity_hs.reshape(it['input_ids'].shape[0], -1, 4, 768) # [batch_size, max_seq_len, entity_pad_len(4), 768]
                    # prompt_entity_hs_fusion = torch.mean(prompt_entity_hs, dim=2) # [batch_size, max_seq_len, 768]
                    prompt_entity_hs_fusion = prompt_entity_hs.reshape(it['input_ids'].shape[0], it['input_ids'].shape[1], -1) # [batch_size, max_seq_len, 4 * 768]

                    # it['prompt_features'] = prompt_features
                    
                    outputs = self.model(**it)
                    hidden_states = outputs['mix_output']

                    loss = self.birnncrf.loss(
                        hidden_states, prompt_entity_hs_fusion, it['input_ids'].gt(0), it['labels'])
                    
                    loss = loss.mean()

                    loss.backward()
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    self.birnncrf.zero_grad()

                    train_loss += loss.data.item()
                    train_count += 1

                    preds = self.birnncrf(hidden_states, prompt_entity_hs_fusion, it['input_ids'].gt(0))[1]

                    for input_ids, pred, labels in zip(it["input_ids"], preds, it["labels"]):
                        remove_pads = input_ids.gt(0)
                        # remove [SEP] and [CLS]
                        pred_labels = pred[1:-1]
                        true_labels = labels[remove_pads][1:-1].tolist()

                        assert len(pred_labels)==len(true_labels)


                        pred_labels = [label.replace(
                            "M-", "I-") for label in self.analysis.idx2tag(pred_labels)]
                        true_labels = [label.replace(
                            "M-", "I-") for label in self.analysis.idx2tag(true_labels)]

                        pred_labels_list.append(pred_labels)
                        true_labels_list.append(true_labels)

                    all_preds += pred_labels_list
                    all_trues += true_labels_list

                    acc = accuracy_score(true_labels_list, pred_labels_list)
                    p = precision_score(
                        true_labels_list, pred_labels_list)
                    r = recall_score(true_labels_list, pred_labels_list)
                    f1 = f1_score(true_labels_list, pred_labels_list)

                    all_acc_list.append(acc)
                    all_p_list.append(p)
                    all_r_list.append(r)
                    all_f1_list.append(f1)

                    train_acc = np.mean(all_acc_list)
                    train_precision = np.mean(all_p_list)
                    train_recall = np.mean(all_r_list)
                    F1 = np.mean(all_f1_list)

                    train_iter.set_description(
                        'Epoch: {}/{} Train'.format(epoch + 1, self.num_epochs))
                    train_iter.set_postfix(train_loss=train_loss / train_count, train_acc=train_acc, train_precision=train_precision,
                                        train_recall=train_recall, F1=F1)

                reports = self.__format_dict(classification_report(
                    all_trues, all_preds, output_dict=True))
                print("train_reports:")
                print(classification_report(all_trues,all_preds))

                self.analysis.append_train_record({
                    'loss': loss.data.item(),
                    'f1': F1,
                    'acc': train_precision,
                    'recall': train_recall,
                    'reports': reports
                })

            model_uid = None
            if save:
                model_uid = self.save_model(train_step)
            if self.eval_data and eval:
                if eval_call_epoch is None or eval_call_epoch(epoch):
                    self.eval()
                else:
                    self.analysis.append_eval_record({
                        'loss': 'skip',
                        'f1': 'skip',
                        'acc': 'skip',
                        'recall': 'skip',
                        'reports': {}
                    })

            if save:
                self.analysis.save_ner_record(
                    uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch + 1, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    def eval(self):

        fct_loss = nn.BCELoss()
        test_count = 0
        eval_loss = 0
        test_iter = tqdm(self.eval_iter)
        self.model.eval()
        self.birnncrf.eval()
        self.prompt_model.eval()
        # [2022-08-29] temp output
        os.makedirs(f"outputs/{self.task_name}/{self.model_name}/",exist_ok=True)
        f = open(f"outputs/{self.task_name}/{self.model_name}/eval_outputs.json","w",encoding="utf-8")
        with torch.no_grad():
            all_acc_list = []
            all_p_list = []
            all_r_list = []
            all_f1_list = []
            all_preds = []
            all_trues = []
            for it in test_iter:
                pred_labels_list = []
                true_labels_list = []

                for key in it.keys():
                    it[key] = self.cuda(it[key])

                # [batch_size, 4, 512]
                prompt_inputs = it['prompt_input_ids']
                prompt_inputs = prompt_inputs.reshape(-1, 512) # [batch_size * 4, 512]
                prompt_origin_labels = it['prompt_origin_labels']
                prompt_origin_labels = prompt_origin_labels.reshape(-1, 512) # [batch_size * 4, 512]
                prompt_attention_mask = prompt_origin_labels.gt(0)
                prompt_indexed = []
                for i in range(it['input_ids'].shape[0]):
                    it['prompt_indexes'][i] = it['prompt_indexes'][i] + 2048 * i
                    prompt_indexed += it['prompt_indexes'][i]
                
                prompt_outputs = self.prompt_model(input_ids=prompt_inputs, attention_mask=prompt_attention_mask)
                prompt_hidden_states = prompt_outputs.last_hidden_state # [batch_size * 4, 512, 768]
                prompt_hidden_states = prompt_hidden_states * prompt_attention_mask.unsqueeze(-1).float() # [batch_size * 4, 512, 768]
                prompt_hidden_states = prompt_hidden_states.reshape(-1, 768) # [batch_size * 4 * 512, 768]
                prompt_entity_hs = prompt_hidden_states[prompt_indexed] # [batch_size * max_seq_len * entity_pad_len(4), 768]
                prompt_entity_hs = prompt_entity_hs.reshape(it['input_ids'].shape[0], -1, 4, 768) # [batch_size, max_seq_len, entity_pad_len(4), 768]
                # prompt_entity_hs_fusion = torch.mean(prompt_entity_hs, dim=2) # [batch_size, max_seq_len, 768]
                prompt_entity_hs_fusion = prompt_entity_hs.reshape(it['input_ids'].shape[0], it['input_ids'].shape[1], -1) # [batch_size, max_seq_len, 768]
                
                outputs = self.model(**it)
                hidden_states = outputs['mix_output']

                loss = self.birnncrf.loss(
                    hidden_states, prompt_entity_hs_fusion, it['input_ids'].gt(0), it['labels'])
                
                loss = loss.mean()

                eval_loss += loss.data.item()
                test_count += 1

                preds = self.birnncrf(hidden_states, prompt_entity_hs_fusion, it['input_ids'].gt(0))[1]

                for input_ids, pred, labels in zip(it["input_ids"], preds, it["labels"]):

                    remove_pads = input_ids.gt(0)
                    # remove [SEP] and [CLS]
                    pred_labels = pred[1:-1]
                    true_labels = labels[remove_pads][1:-1].tolist()

                    assert len(pred_labels)==len(true_labels)

                    pred_labels = [label.replace(
                        "M-", "I-") for label in self.analysis.idx2tag(pred_labels)]
                    true_labels = [label.replace(
                        "M-", "I-") for label in self.analysis.idx2tag(true_labels)]

                    pred_labels_list.append(pred_labels)
                    true_labels_list.append(true_labels)
                    # output predicted answer
                    f.write(json.dumps({
                        "pred": pred_labels,
                        "true": true_labels,
                        "input_ids": input_ids[remove_pads][1:-1].tolist()
                    }))
                    f.write("\n")

                # input ids

                all_preds += pred_labels_list
                all_trues += true_labels_list

                acc = accuracy_score(true_labels_list, pred_labels_list)
                p = precision_score(
                    true_labels_list, pred_labels_list)
                r = recall_score(true_labels_list, pred_labels_list)
                f1 = f1_score(true_labels_list, pred_labels_list)

                all_acc_list.append(acc)
                all_p_list.append(p)
                all_r_list.append(r)
                all_f1_list.append(f1)

                test_acc = np.mean(all_acc_list)
                test_precision = np.mean(all_p_list)
                test_recall = np.mean(all_r_list)
                F1 = np.mean(all_f1_list)

                test_iter.set_description('Eval Result')
                test_iter.set_postfix(
                    eval_loss=eval_loss / test_count, eval_acc=test_acc, eval_precision=test_precision, eval_recall=test_recall, F1=F1)

            # close file
            f.close()

            reports = self.__format_dict(classification_report(
                all_trues, all_preds, output_dict=True))
            print("eval_reports:")
            print(classification_report(all_trues,all_preds))
            
            self.analysis.append_eval_record({
                'loss': loss.data.item(),
                'f1': F1,
                'acc': test_precision,
                'recall': test_recall,
                "reports": reports
            })
    
    def __format_dict(self,d):
        if isinstance(d,dict):
            for k in d:
                d[k] = self.__format_dict(d[k])
            return d
        else:
            return d.item()

    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists('./save_model/{}/{}'.format(dir, self.model_name)):
            os.makedirs('./save_model/{}/{}'.format(dir, self.model_name))
        if not os.path.exists('./save_model/{}/lstm_crf'.format(dir)):
            os.makedirs('./save_model/{}/lstm_crf'.format(dir))
        torch.save(
            self.model, './save_model/{}/{}/{}_{}.pth'.format(dir, self.model_name, self.model_name, current_step))
        torch.save(self.birnncrf,
                   './save_model/{}/lstm_crf/lstm_crf_{}.pth'.format(dir, current_step))
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
