import re
import torch
import random
from CC.loaders.utils import DataManager
from ICCSupervised.ICCSupervised import IDataLoader
from torch.utils.data import DataLoader, Dataset


class CNDataLoader(IDataLoader):
    def __init__(self, **args):
        assert "train_file" in args, "argument train_file required"
        assert "bert_vocab_file" in args, "argument bert_vocab_file required"
        assert "tag_file" in args, "argument tag_file required"
        
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
        self.myData = CNDataset(
            self.train_set, self.train_tags, self.dm, padding_length, self.do_shuffle)
        self.dataiter = DataLoader(self.myData, batch_size=batch_size)
        if self.output_eval:
            self.myData_eval = CNDataset(
                self.eval_set, self.eval_tags, self.dm, padding_length)
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


class CNDataset(Dataset):
    def __init__(self, sentences_list, tags_list, data_manager, padding_length=100, do_shuffle=False):
        self.sentences, self.tags_list = sentences_list, tags_list
        self.data_manager = data_manager
        self.padding_length = padding_length
        self.do_shuffle = do_shuffle

        self.shuffle_idx_list = [idx for idx in range(len(self.sentences))]
        if self.do_shuffle:
            random.shuffle(self.shuffle_idx_list)
            

    def __getitem__(self, idx):
        idx = self.shuffle_idx_list[idx]
        sentence, tags = self.data_manager.encode(
            self.sentences[idx], self.tags_list[idx], padding_length=self.padding_length - 1)
        sentence = [101] + sentence
        tags = [self.data_manager.tag_to_idx['O']] + tags
        for i, _ in enumerate(sentence):
            if _ == 0:
                sentence[i] = 102
                tags[i] = self.data_manager.tag_to_idx['O']
                break
        sentence = torch.tensor(sentence)
        tags = torch.tensor(tags)
        return {
            'input_ids': sentence,
            'attention_mask': sentence.gt(0),
            'labels': tags
        }

    def __len__(self):
        return len(self.sentences)
