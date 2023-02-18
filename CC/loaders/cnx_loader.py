import re
import torch
import random
from CC.loaders.utils import DataManager
from ICCSupervised.ICCSupervised import IDataLoader
from torch.utils.data import DataLoader, Dataset


class CNXDataLoader(IDataLoader):
    def __init__(self, **args):
        assert "train_file" in args, "argument train_file required"
        assert "bert_vocab_file" in args, "argument bert_vocab_file required"
        assert "tag_file" in args, "argument tag_file required"
        assert "tag_rules" in args, "argument tag_rules required"

        self.output_eval = False
        if "eval_file" in args:
            self.output_eval = True

        self.word_tag_split = ' '
        if "word_tag_split" in args:
            self.word_tag_split = args['word_tag_split']

        self.eval_file = None
        if "eval_file" in args:
            self.eval_file = args['eval_file']

        self.pattern = '， O'
        if "pattern" in args:
            self.pattern = args['pattern']

        self.max_seq_length = 50
        if "max_seq_length" in args:
            self.max_seq_length = args['max_seq_length']

        self.use_json = False
        if "use_json" in args:
            self.use_json = args['use_json']

        self.batch_size = 32
        if "batch_size" in args:
            self.batch_size = args['batch_size']

        self.eval_batch_size = None
        if "eval_batch_size" in args:
            self.eval_batch_size = args['eval_batch_size']
        
        self.do_shuffle = False
        if "do_shuffle" in args:
            self.do_shuffle = args['do_shuffle']

        self.tag_rules = args['tag_rules']
        self.read_data_set(args['train_file'], self.eval_file,
                           self.word_tag_split, self.pattern)
        self.data_manager_init(args['bert_vocab_file'], args['tag_file'])
        self.verify_data()
        self.process_data(self.max_seq_length,
                          self.batch_size, self.eval_batch_size)

    '''
    全局数据控制器
    bert_vocab_file: 单词表文件名
    tag_file: 标签表文件名
    '''

    def data_manager_init(self, bert_vocab_file, tag_file):
        tags_list = DataManager.ReadTagsList(tag_file)
        self.dm = DataManager(tags_list=tags_list,
                              vocab_file_name=bert_vocab_file)

    '''
    读取数据
    '''

    def read_data_set(self, train_file, eval_file=None, word_tag_split=' ', pattern='， O'):
        if self.use_json:
            self.train_set, self.train_tags = DataManager.ReadJsonData(
                train_file)
            if eval_file is not None:
                self.eval_set, self.eval_tags = DataManager.ReadJsonData(
                    eval_file)
        else:
            self.train_set, self.train_tags = DataManager.ReadDataExtremely(
                train_file, word_tag_split, pattern)
            if eval_file is not None:
                self.eval_set, self.eval_tags = DataManager.ReadDataExtremely(
                    eval_file, word_tag_split, pattern)

    '''
    验证数据
    '''

    def verify_data(self):
        if len(self.train_set) != len(self.train_tags):
            raise Exception(
                "The number of train_sentence is not equal to train_tags.")
        for i, item in enumerate(self.train_set):
            if len(self.train_set[i]) != len(self.train_tags[i]):
                raise Exception(
                    "The length of sentence at index: {} is not equal to the length of corresponding tags.".format(i))
            for tag in self.train_tags[i]:
                if tag not in self.dm.tag_to_idx:
                    raise Exception(
                        "The tag of '{}' is not exists in tags_list.".format(tag))

    '''
    处理数据
    '''

    def process_data(self, padding_length, batch_size, eval_batch_size):
        eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
        self.myData = CNXDataset(
            self.train_set, self.train_tags, self.dm, self.tag_rules, padding_length, self.do_shuffle)
        self.dataiter = DataLoader(self.myData, batch_size=batch_size)
        if self.output_eval:
            self.myData_eval = CNXDataset(
                self.eval_set, self.eval_tags, self.dm, self.tag_rules, padding_length)
            self.dataiter_eval = DataLoader(
                self.myData_eval, batch_size=eval_batch_size)

    def __call__(self):
        if self.output_eval:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'eval_set': self.myData_eval,
                'eval_iter': self.dataiter_eval,
                'dm': self.dm
            }
        else:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'dm': self.dm
            }


class CNXDataset(Dataset):
    def __init__(self, sentences_list, tags_list, data_manager, tag_rules, padding_length=100, do_shuffle=False):
        self.sentences, self.tags_list = sentences_list, tags_list
        self.tag_rules = tag_rules
        self.data_manager = data_manager
        self.padding_length = padding_length
        self.generate_prompt()

        self.do_shuffle = do_shuffle

        self.shuffle_idx_list = [idx for idx in range(len(self.sentences))]
        if self.do_shuffle:
            random.shuffle(self.shuffle_idx_list)

    def generate_prompt(self):
        prompt_inputs = []
        prompt_labels = []
        prompt_origin_labels = []
        for idx, tags in enumerate(self.tags_list):
            sentence = self.sentences[idx]
            prompt_input = []
            prompt_label = []
            prompt_origin_label = []
            range_text = ''
            range_tag = ''
            for i, tag in enumerate(tags):
                s = tag.split('-')[0]
                if s == 'B' or s == 'S':
                    e = tag.split('-')[1]
                    if e not in self.tag_rules:
                        raise('{} is not in the tag_rules'.format(e))
                    range_tag = self.tag_rules[e]
                    range_text += sentence[i]
                elif s == 'I' or s == 'E':
                    range_text += sentence[i]
                elif s == 'O':
                    if range_text == '':
                        continue
                    prompt_input += [_ for _ in range_text] + \
                        ['是', '一', '个'] + ['[MASK]' for _ in range_tag] + [';']
                    prompt_label += [word for word in range_text] + \
                        [-100, -100, -100] + [word for word in range_tag] + [-100]
                    prompt_origin_label += [word for word in range_text] + \
                        ['是', '一', '个'] + [word for word in range_tag] + [';']
                    range_text = ''
                    range_tag = ''
            prompt_inputs.append(prompt_input)
            prompt_labels.append(prompt_label)
            prompt_origin_labels.append(prompt_origin_label)
        self.prompt_inputs = prompt_inputs
        self.prompt_labels = prompt_labels
        self.prompt_origin_labels = prompt_origin_labels

    def __getitem__(self, idx):
        idx = self.shuffle_idx_list[idx]
        input_ids = self.sentences[idx] + ['[SEP]'] + self.prompt_inputs[idx]
        input_ids = [self.data_manager.wordToIdx(word) for word in input_ids]
        input_ids = [101] + input_ids

        labels = [-100 for _ in input_ids]
        if len(self.prompt_labels[idx]) > 0:
            labels[-len(self.prompt_labels[idx]):] = self.prompt_labels[idx]
        labels = [self.data_manager.wordToIdx(word) if word != -100 else -100 for word in labels]

        origin_labels = [_ for _ in input_ids]
        if len(self.prompt_origin_labels[idx]) > 0:
            prompt_origin_labels = [self.data_manager.wordToIdx(word) for word in self.prompt_origin_labels[idx]]
            origin_labels[-len(prompt_origin_labels):] = prompt_origin_labels

        token_type_ids = [0 for _ in input_ids]
        for i in range(len(self.sentences[idx]) + 1, len(token_type_ids)):
            token_type_ids[i] = 1
        if len(input_ids) > self.padding_length:
            input_ids = input_ids[:self.padding_length]
            labels = labels[:self.padding_length]
            origin_labels = origin_labels[:self.padding_length]
            token_type_ids = token_type_ids[:self.padding_length]
        else:
            remain = self.padding_length - len(input_ids)
            for i in range(remain):
                input_ids.append(self.data_manager.wordToIdx('[PAD]'))
                labels.append(-100)
                origin_labels.append(-100)
                token_type_ids.append(1)
        
        sentence = torch.tensor(input_ids)
        tags = torch.tensor(labels)
        origin_tags = torch.tensor(origin_labels)
        token_type_ids = torch.tensor(token_type_ids)
        return {
            'input_ids': sentence,
            'attention_mask': sentence.gt(0),
            'token_type_ids': token_type_ids,
            'input_labels': tags,
            'origin_labels': origin_tags,
            'labels': tags
        }

    def __len__(self):
        return len(self.sentences)
