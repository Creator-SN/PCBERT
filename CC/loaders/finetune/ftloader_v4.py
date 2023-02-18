import collections
import enum
from os import read
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
import math


class FTLoaderV4(IDataLoader):
    def __init__(self, **args):
        KwargsParser(debug=True) \
            .add_argument("batch_size", int, defaultValue=4) \
            .add_argument("eval_batch_size", int, defaultValue=64) \
            .add_argument("test_batch_size", int, defaultValue=64) \
            .add_argument("train_file", str) \
            .add_argument("eval_file", str) \
            .add_argument("test_file", str) \
            .add_argument("tag_file", str) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("word_vocab_file", str) \
            .add_argument("word_embedding_file", str) \
            .add_argument("max_seq_length", int, defaultValue=256) \
            .add_argument("do_shuffle", bool, defaultValue=False) \
            .add_argument("task_name", str) \
            .add_argument("tag_rules", dict) \
            .add_argument("debug", bool, defaultValue=False) \
            .add_argument("pass_none_rule", bool, defaultValue=False) \
            .add_argument("default_tag", str, defaultValue="O") \
            .add_argument("max_scan_num", int, defaultValue=1000000) \
            .add_argument("max_word_num", int, defaultValue=5) \
            .add_argument("add_seq_vocab", bool, defaultValue=False) \
            .add_argument("use_test", bool, defaultValue=False) \
            .add_argument("output_eval", bool, defaultValue=True) \
            .parse(self, **args)

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
        self.process_data()

    def read_data_set(self):

        self.data_files: List[str] = [
            self.train_file, self.eval_file, self.test_file
        ]

        cache = self.cache.group(self.max_scan_num)

        self.lexicon_tree = cache.load(
            "lexicon_tree", lambda: TrieFactory.get_trie_from_vocabs(
                [self.word_vocab_file], self.max_scan_num))

        self.matched_words = cache.load(
            "matched_words",
            lambda: TrieFactory.get_all_matched_word_from_dataset(
                self.data_files, self.lexicon_tree))

        self.word_vocab: Vocab = cache.load(
            "word_vocab", lambda: Vocab().from_list(
                self.matched_words, is_word=True, has_default=False, unk_num=5)
        )

        self.tag_vocab: Vocab = Vocab().from_files([self.tag_file],
                                                   is_word=False)

        self.vocab_embedding, self.embedding_dim = cache.load(
            "vocab_embedding",
            lambda: VocabEmbedding(self.word_vocab).build_from_file(
                self.word_embedding_file, self.max_scan_num, self.add_seq_vocab
            ).get_embedding())

        self.max_tag_length = max(len(i) for i in self.tag_rules.values())

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

    def process_data(self):
        if self.use_test:
            self.myData_test = FTLoaderV4DataSet(
                self.test_file, self.tokenizer, self.tag_rules,
                self.max_seq_length, self.max_tag_length, self.default_tag,
                self.tag_vocab, self.lexicon_tree, self.max_word_num,
                self.word_vocab)
            self.dataiter_test = DataLoader(self.myData_test,
                                            batch_size=self.test_batch_size)
        else:
            self.myData = FTLoaderV4DataSet(self.train_file,
                                            self.tokenizer,
                                            self.tag_rules,
                                            self.max_seq_length,
                                            self.max_tag_length,
                                            self.default_tag,
                                            self.tag_vocab,
                                            self.lexicon_tree,
                                            self.max_word_num,
                                            self.word_vocab)
            self.dataiter = DataLoader(self.myData, batch_size=self.batch_size,
                                       shuffle=self.do_shuffle)
            if self.output_eval:
                self.myData_eval = FTLoaderV4DataSet(
                    self.eval_file,
                    self.tokenizer,
                    self.tag_rules,
                    self.max_seq_length,
                    self.max_tag_length,
                    self.default_tag,
                    self.tag_vocab,
                    self.lexicon_tree,
                    self.max_word_num,
                    self.word_vocab,
                )
                self.dataiter_eval = DataLoader(
                    self.myData_eval, batch_size=self.eval_batch_size)

    def __call__(self):
        if not self.use_test:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'eval_set': self.myData_eval,
                'eval_iter': self.dataiter_eval,
                'lexicon_tree': self.lexicon_tree,
                'tag_vocab': self.tag_vocab,
                'word_vocab': self.word_vocab,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim
            }
        else:
            return {
                'test_set': self.myData_test,
                'test_iter': self.dataiter_test,
                'lexicon_tree': self.lexicon_tree,
                'tag_vocab': self.tag_vocab,
                'word_vocab': self.word_vocab,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim
            }


class FTLoaderV4DataSet(Dataset):
    def __init__(self,
                 file: str,
                 tokenizer: BertTokenizer,
                 tag_rules: Dict[str, str],
                 max_seq_length: int,
                 max_tag_length: int,
                 default_tag: str,
                 label_vocab: Vocab,
                 lexicon_tree: Trie,
                 max_word_num: int,
                 word_vocab: Vocab,
                 do_shuffle: bool = False):
        self.__dict__.update(locals().items())
        self.__init_dataset()

    def convert_prompts(self, item):
        if "text" not in item:
            raise KeyError(f"key text not exists in item: {item}")
        if "label" not in item:
            raise KeyError(f"key label not exists in item: {item}")
        text, label = item["text"][:self.max_seq_length - 2], item["label"]

        ids = self.tokenizer(''.join(text))

        cur_length = len(ids["input_ids"])
        offset = self.max_seq_length
        padding = self.max_tag_length + 2

        max_padding = 512

        text_ids = {
            "prompt_input_ids": torch.zeros(max_padding, dtype=torch.int),
            "prompt_attention_mask": torch.zeros(max_padding, dtype=torch.int),
            "prompt_token_type_ids": torch.zeros(max_padding, dtype=torch.int),
            "prompt_labels": torch.tensor([-100] * max_padding,
                                          dtype=torch.int),
            "prompt_origin_labels": None,
        }
        text_ids["prompt_input_ids"][:cur_length] = torch.tensor(
            ids["input_ids"], dtype=torch.int)

        text_ids["prompt_attention_mask"][:cur_length] = 1

        text_ids["prompt_token_type_ids"][cur_length:] = 1

        text_ids["prompt_origin_labels"] = text_ids["prompt_input_ids"].clone()

        current = dict(
            zip(text_ids.keys(), [i.clone() for i in text_ids.values()]))

        collections = {}
        for key in current.keys():
            collections[key] = []

        # [512 / 7] * 4
        index_length = math.ceil(
            (max_padding - self.max_seq_length) /
            (3 + self.max_tag_length)) * self.max_tag_length
        # index_length = 300

        indexes = torch.zeros(4 * self.max_seq_length, dtype=np.int)
        index_offset = 0
        cur_length = offset
        count = 0
        for i in range(self.max_seq_length):
            prompt_text = list(f"{i}是") + ["[MASK]"] * self.max_tag_length + [
                ","
            ]
            if cur_length + len(prompt_text) >= max_padding:
                count += 1
                # 超出长度判断，生成当前组数据
                for key in collections.keys():
                    collections[key].append(current[key])
                # 重置数据，为下一组数据做准备
                cur_length = offset
                current = dict(
                    zip(text_ids.keys(),
                        [i.clone() for i in text_ids.values()]))
            # 左闭右开
            start, end = cur_length + len(str(i)) + 1, cur_length + len(
                str(i)) + padding - 1

            for j in range(start, end):
                indexes[index_offset] = count * max_padding + j
                index_offset += 1

            char_label = self.tag_rules[
                label[i - 1].split("-")
                [-1]] if i != 0 and i - 1 < len(text) else self.tag_rules["O"]

            current["prompt_input_ids"][cur_length:cur_length +
                                        len(prompt_text)] = torch.tensor(
                                            self.tokenizer.
                                            convert_tokens_to_ids(prompt_text),
                                            dtype=torch.int)

            current["prompt_attention_mask"][cur_length:cur_length +
                                             len(prompt_text)] = torch.tensor(
                                                 [1] * (len(str(i)) + 1) +
                                                 [0] * self.max_tag_length +
                                                 [1],
                                                 dtype=torch.int)
            mask_label_ids = self.tokenizer.convert_tokens_to_ids(
                list(char_label))
            current["prompt_labels"][cur_length:cur_length +
                                     len(prompt_text)] = torch.tensor(
                                         [-100] * (len(str(i)) + 1) +
                                         mask_label_ids + [0] *
                                         (self.max_tag_length -
                                          len(mask_label_ids)) + [-100],
                                         dtype=torch.int)

            current["prompt_origin_labels"][
                cur_length:cur_length + len(prompt_text)] = torch.tensor(
                    self.tokenizer.convert_tokens_to_ids(list(str(i) + '是')) +
                    mask_label_ids + [0] *
                    (self.max_tag_length - len(mask_label_ids)) +
                    self.tokenizer.convert_tokens_to_ids([","]))

            cur_length += len(prompt_text)

        # 最后一组数据
        for key in collections.keys():
            collections[key].append(current[key].clone())

        # 按序合并到一维
        for key in collections.keys():
            collections[key] = torch.stack(collections[key]).reshape(-1)

        collections["prompt_indexes"] = indexes

        return collections

    def convert_embedding(self, obj, return_dict: bool = False):
        if "text" not in obj:
            raise ValueError("obj required attribute: text")
        text = ["[CLS]"] + obj["text"][:self.max_seq_length - 2] + ["[SEP]"]
        if "label" not in obj and self.do_predict:
            label = [self.default_tag] * self.max_seq_length
        elif "label" not in obj:
            raise ValueError("obj required attribute: label")
        else:
            label = [self.default_tag] + \
                obj["label"][:self.max_seq_length-2]+[self.default_tag]
        # convert to embedding
        token_ids = self.tokenizer.convert_tokens_to_ids(text)
        label_ids = self.label_vocab.token2id(label)

        labels = torch.zeros(self.max_seq_length, dtype=torch.int)
        labels[:len(label_ids)] = tensor(label_ids[:self.max_seq_length]).int()
        # init input
        input_token_ids = torch.zeros(self.max_seq_length, dtype=torch.int)
        input_token_ids[:len(token_ids)] = tensor(
            token_ids[:self.max_seq_length]).int()
        token_type_ids = torch.ones(self.max_seq_length, dtype=torch.int)
        token_type_ids[:len(token_ids)] = 0

        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.int)
        attention_mask[:len(token_ids)] = 1

        matched_word_ids = torch.zeros(
            (self.max_seq_length, self.max_word_num), dtype=torch.int)
        matched_word_mask = torch.zeros(
            (self.max_seq_length, self.max_word_num), dtype=torch.int)

        # get matched word
        matched_words = self.lexicon_tree.getAllMatchedWordList(
            text, self.max_word_num)
        for i, words in enumerate(matched_words):
            word_ids = self.word_vocab.token2id(words)
            matched_word_ids[i][:len(word_ids)] = tensor(word_ids,
                                                         dtype=torch.int)
            matched_word_mask[i][:len(word_ids)] = 1

        if return_dict:
            return {
                "input_ids": input_token_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "matched_word_ids": matched_word_ids,
                "matched_word_mask": matched_word_mask,
                "labels": labels,
            }

        return input_token_ids, attention_mask, token_type_ids, matched_word_ids, matched_word_mask, labels

    def __init_dataset(self):
        reader = FileReader(self.file)
        line_total = reader.line_size()

        self.dataset = []
        # left model keys
        self.keys = {
            "input_ids", "attention_mask", "token_type_ids",
            "matched_word_ids", "matched_word_mask", "labels"
        }
        # right model keys
        self.prompt_keys = [
            "input_ids", "attention_mask", "token_type_ids", "labels",
            "origin_labels", "indexes"
        ]
        self.prompt_keys = [f"prompt_{i}" for i in self.prompt_keys]

        for line in tqdm(reader.line_iter(),
                         desc=f"load dataset from {self.file}",
                         total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)

            d = self.convert_embedding(data, return_dict=True)
            d.update(self.convert_prompts(data))

            self.dataset.append(d)

        self.size = len(self.dataset)
        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        if isinstance(idx, list):
            data = {}
            for key in self.prompt_keys:
                data[key] = [self.dataset[j][key] for j in idx]
                data[key] = torch.stack(data[key])

            for key in self.keys:
                data[key] = [self.dataset[j][key] for j in idx]
                data[key] = torch.stack(data[key])
            return data
        return self.dataset[idx]

    def __len__(self):
        return self.size
