from functools import lru_cache
import json
from typing import Any, Dict, List, Union
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.utils.dummy_pt_objects import BertModel
from CC.loaders.utils.cache_manager import FileCache
from CC.loaders.utils.embedding import VocabEmbedding
from CC.loaders.utils.label import get_entities
from CC.loaders.utils.lexicon_factory import TrieFactory
from CC.loaders.utils.lexicon_tree import Trie
from CC.loaders.utils.parser import KwargsParser
from CC.loaders.utils.reader import FileReader
from CC.loaders.utils.vocab import Vocab
from transformers import BertTokenizer, BertModel
from ICCSupervised.ICCSupervised import IDataLoader
import random
from torch import tensor


class PTLoaderV1(IDataLoader):
    """ Pretrain Loader Version 1
    feature:
        positive sample and negative sample, others sample
        GPT2 loader

    """
    def __init__(self, **kwargs) -> None:
        KwargsParser(debug=True) \
            .add_argument("batch_size", int, defaultValue=4) \
            .add_argument("eval_batch_size", int, defaultValue=16) \
            .add_argument("test_batch_size", int, defaultValue=16) \
            .add_argument("word_embedding_file", str) \
            .add_argument("word_vocab_file", str) \
            .add_argument("train_file", str) \
            .add_argument("eval_file", str) \
            .add_argument("test_file", str) \
            .add_argument("tag_file", str) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("output_eval", bool, defaultValue=False) \
            .add_argument("max_seq_length", int, defaultValue=256) \
            .add_argument("default_tag", str, defaultValue="O") \
            .add_argument("use_test", bool, defaultValue=False) \
            .add_argument("do_shuffle", bool, defaultValue=False) \
            .add_argument("do_predict", bool, defaultValue=False) \
            .add_argument("task_name", str) \
            .add_argument("debug", bool, defaultValue=False) \
            .parse(self, **kwargs)

        # get cache_key
        files = [
            self.train_file, self.eval_file, self.test_file, self.tag_file
        ]
        self.cache_key = [
            FileReader(file).etag() if file is not None else "None"
            for file in files
        ]
        self.cache_key = "_".join(self.cache_key)
        self.cache = FileCache(f"./temp/{self.cache_key}")

        self.read_data_set()
        self.process_data(self.batch_size, self.eval_batch_size,
                          self.test_batch_size)

    def read_data_set(self):
        self.data_files: List[str] = [
            self.train_file, self.eval_file, self.test_file
        ]

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

        self.tag_vocab: Vocab = Vocab().from_files([self.tag_file],
                                                   is_word=False)

    def process_data(self,
                     batch_size: int,
                     eval_batch_size: int = None,
                     test_batch_size: int = None):
        if self.use_test:
            self.myData_test = PTDataSetV1(self.data_files[2],
                                           tokenizer=self.tokenizer,
                                           max_seq_length=self.max_seq_length,
                                           label_vocab=self.tag_vocab,
                                           default_tag=self.default_tag)
            self.dataiter_test = DataLoader(self.myData_test,
                                            batch_size=test_batch_size)
        else:
            self.myData = PTDataSetV1(self.data_files[0],
                                      tokenizer=self.tokenizer,
                                      max_seq_length=self.max_seq_length,
                                      label_vocab=self.tag_vocab,
                                      default_tag=self.default_tag,
                                      do_shuffle=True)

            self.dataiter = DataLoader(self.myData, batch_size=batch_size)
            if self.output_eval:
                self.myData_eval = PTDataSetV1(
                    self.data_files[1],
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_seq_length,
                    label_vocab=self.tag_vocab,
                    default_tag=self.default_tag)
                self.dataiter_eval = DataLoader(self.myData_eval,
                                                batch_size=eval_batch_size)

    def __call__(self):
        if self.use_test:
            return {
                'test_set': self.myData_test,
                'test_iter': self.dataiter_test,
                'tag_vocab': self.tag_vocab
            }
        if self.output_eval:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'eval_set': self.myData_eval,
                'eval_iter': self.dataiter_eval,
                'tag_vocab': self.tag_vocab
            }
        else:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'tag_vocab': self.tag_vocab
            }


class PTDataSetV1(Dataset):
    def __init__(self,
                 file: str,
                 tokenizer,
                 max_seq_length: int,
                 label_vocab,
                 default_tag: str = "O",
                 do_shuffle: bool = False,
                 do_predict: bool = False,
                 **kwargs) -> None:
        params = locals()
        for key in params:
            if key != "kwargs":
                setattr(self, key, params[key])
        if not self.do_predict:
            self.init_dataset()

    def convert_ids(self, obj):
        text, labels = obj["text"][:self.max_seq_length -
                                   2], obj["label"][:self.max_seq_length - 2]

        prompt = text[:]
        for index in range(len(prompt)):
            id = self.label_vocab.token2id(labels[index])
            prompt[index] = f"[unused{id + 1}]"

        input_ids = torch.zeros(self.max_seq_length, dtype=torch.int)
        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.int)
        token_type_ids = torch.zeros(self.max_seq_length, dtype=torch.int)
        input_labels =  torch.zeros(self.max_seq_length, dtype=torch.int)
        text = ["[CLS]"] + text + ["[SEP]"]
        input_ids[:len(text)] = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(text)).int()
        input_labels[:len(prompt)] = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(prompt)).int()
        attention_mask[:len(text)] = 1

        # token_type_ids[len(text):] = 1

        # ids = self.tokenizer.encode_plus(text,
        #                                  prompt,
        #                                  max_length=self.max_seq_length,
        #                                  truncation="only_second",
        #                                  padding="max_length",
        #                                  return_tensors="pt")

        # input_labels = torch.clone(input_ids)

        # for i in range(token_type_ids.shape[0]):
        #     if token_type_ids[i] == 0:
        #         input_labels[i] = -100

        # for i in range(len(text) - 1, len(text) + len(prompt) - 1):
        #     input_labels[i] = input_labels[i + 1]
        # input_labels[len(text) + len(prompt) - 1] = -100

        input_labels[len(prompt):] = -100

        return input_ids, token_type_ids, attention_mask, input_labels

    def init_dataset(self):
        reader = FileReader(self.file)
        line_total = reader.line_size()
        self.input_token_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.input_labels = []

        for line in tqdm(reader.line_iter(),
                         desc=f"load dataset from {self.file}",
                         total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            if len(data["text"]) > 0:
                input_token_ids, token_type_ids, attention_mask, input_labels = self.convert_ids(
                    data)
                self.input_token_ids.append(input_token_ids)
                self.token_type_ids.append(token_type_ids)
                self.attention_mask.append(attention_mask)
                self.input_labels.append(input_labels)

        self.size = len(self.input_token_ids)
        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, index):
        idx = self.indexes[index]
        if isinstance(idx, list):
            return {
                'input_ids':
                torch.stack([self.input_token_ids[i] for i in idx]),
                'attention_mask':
                torch.stack([self.attention_mask[i] for i in idx]),
                'token_type_ids':
                torch.stack([self.token_type_ids[i] for i in idx]),
                "input_labels":
                torch.stack([self.input_labels[i] for i in idx])
            }
        else:
            return {
                'input_ids': self.input_token_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'token_type_ids': self.token_type_ids[idx],
                "input_labels": self.input_labels[idx]
            }

    def __len__(self):
        return self.size
