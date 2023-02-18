import torch
from torch._C import dtype
from CC.loaders.utils import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor
from transformers import BertTokenizer
from tqdm import *
from typing import *
from CC.loaders.utils.label import get_labels
from ICCSupervised.ICCSupervised import IDataLoader
import json
import numpy as np
import random


class PTLoaderV2(IDataLoader):
    def __init__(self, **args):
        KwargsParser(debug=True) \
            .add_argument("batch_size", int, defaultValue=4) \
            .add_argument("train_file", str) \
            .add_argument("tag_file", str) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("max_seq_length", int, defaultValue=256) \
            .add_argument("do_shuffle", bool, defaultValue=False) \
            .add_argument("task_name", str) \
            .add_argument("tag_rules", dict) \
            .add_argument("debug", bool, defaultValue=False) \
            .add_argument("pass_none_rule", bool, defaultValue=False) \
            .parse(self, **args)

        self.read_data_set()
        self.process_data()

    def read_data_set(self):

        self.max_tag_length = max(len(i) for i in self.tag_rules.values())

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

    def process_data(self):
        self.myData = PTLoaderV2DataSet(self.train_file,
                                        self.tokenizer,
                                        self.tag_rules,
                                        self.max_seq_length,
                                        self.max_tag_length,
                                        do_shuffle=self.do_shuffle)
        self.dataiter = DataLoader(self.myData, batch_size=self.batch_size)

    def __call__(self):
        return {'train_set': self.myData, 'train_iter': self.dataiter}


class PTLoaderV2DataSet(Dataset):
    def __init__(self,
                 file: str,
                 tokenizer: BertTokenizer,
                 tag_rules: Dict[str, str],
                 max_seq_length: int,
                 max_tag_length: int,
                 do_shuffle: bool = False):
        self.__dict__.update(locals().items())
        self.__init_dataset()

    def convert_embedding(self, item):
        if "text" not in item:
            raise KeyError(f"key text not exists in item: {item}")
        if "label" not in item:
            raise KeyError(f"key label not exists in item: {item}")
        text, label = item["text"][:self.max_seq_length - 2], item["label"]

        choices_index = set()

        prompts = []
        prompts_attention_mask = []
        prompts_labels = []
        prompts_origin_labels = []

        cur_length = len(text) + 2
        padding = self.max_tag_length + 2
        # 优先选取边界
        for i in range(len(text)):
            if cur_length + 2 * padding + len(str(i)) + len(
                    str(i + 1)) < self.max_seq_length and i + 1 < len(
                        text) and label[i].split("-")[-1]!=label[i + 1].split("-")[-1]:
                if i + 1 not in choices_index:
                    choices_index.add(i + 1)
                    cur_length += padding + len(str(i + 1))
                if i + 2 not in choices_index:
                    choices_index.add(i + 2)
                    cur_length += padding + len(str(i + 2))
        # 按素数步进选取
        for i in range(0, len(text), 3):
            if cur_length + padding + len(str(i)) < self.max_seq_length:
                if i + 1 not in choices_index:
                    choices_index.add(i + 1)
                    cur_length += padding + len(str(i + 1))
            else:
                break
        choices_index = sorted(choices_index)
        # 生成prompts
        for i in choices_index:
            prompts += self.tokenizer.convert_tokens_to_ids(
                list(str(i)) + ["是"] + ["[MASK]"] * self.max_tag_length +
                [","])
            prompts_attention_mask += [1] * (
                len(str(i)) + 1) + [0] * self.max_tag_length + [1]
            mask_ids = self.tokenizer.convert_tokens_to_ids(
                list(self.tag_rules[label[i-1].split("-")[-1]]))
            prompts_labels += [-100] * (len(str(i)) + 1) + mask_ids + [0] * (
                self.max_tag_length - len(mask_ids)) + [-100]
            prompts_origin_labels += self.tokenizer.convert_tokens_to_ids(
                list(str(i) + '是')) + mask_ids + [0] * (
                    self.max_tag_length -
                    len(mask_ids)) + self.tokenizer.convert_tokens_to_ids(
                        [","])

        # 长度检查
        assert len(prompts) == len(prompts_attention_mask)
        assert len(prompts_labels) == len(prompts_origin_labels)
        assert len(prompts) == len(prompts_labels)

        ids = self.tokenizer(''.join(text))
        input_ids = torch.zeros(self.max_seq_length, dtype=torch.int)
        text_length = len(ids["input_ids"])
        input_ids[:text_length] = torch.tensor(ids["input_ids"],
                                               dtype=torch.int)
        origin_labels = input_ids.clone()

        input_ids[text_length:text_length + len(prompts)] = torch.tensor(
            prompts, dtype=torch.int)
        origin_labels[text_length:text_length +
                      len(prompts_origin_labels)] = torch.tensor(
                          prompts_origin_labels, dtype=torch.int)

        labels = torch.tensor([-100] * self.max_seq_length, dtype=torch.int)
        labels[text_length:text_length + len(prompts_labels)] = torch.tensor(
            prompts_labels, dtype=torch.int)

        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.int)
        attention_mask[:text_length] = torch.tensor(
            ids["attention_mask"]).int()
        attention_mask[text_length:text_length +
                       len(prompts_attention_mask)] = torch.tensor(
                           prompts_attention_mask, dtype=torch.int)

        token_type_ids = torch.zeros(self.max_seq_length, dtype=torch.int)
        token_type_ids[text_length:] = 1

        return input_ids, attention_mask, token_type_ids, labels, origin_labels

    def __init_dataset(self):
        line_total = FileUtil.count_lines(self.file)

        self.dataset = []
        self.keys = [
            "input_ids", "attention_mask", "token_type_ids", "labels",
            "origin_labels"
        ]

        for line in tqdm(FileUtil.line_iter(self.file),
                         desc=f"load dataset from {self.file}",
                         total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            self.dataset.append(self.convert_embedding(data))

        self.size = len(self.dataset)
        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        if isinstance(idx, list):
            return dict(
                zip(self.keys, [
                    torch.stack([self.dataset[i][j] for i in idx])
                    for j in range(len(self.keys))
                ]))
        return dict(zip(self.keys, self.dataset[idx]))

    def __len__(self):
        return self.size