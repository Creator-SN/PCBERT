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


class FTLoaderV2(IDataLoader):
    """Fine-Tune Loader Version 2
    feature:
        positive sample and negative sample
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
            .add_argument("max_scan_num", int, defaultValue=1000000) \
            .add_argument("add_seq_vocab", bool, defaultValue=False) \
            .add_argument("max_seq_length", int, defaultValue=256) \
            .add_argument("max_word_num", int, defaultValue=5) \
            .add_argument("max_label_num",int,defaultValue=5) \
            .add_argument("default_tag", str, defaultValue="O") \
            .add_argument("use_test", bool, defaultValue=False) \
            .add_argument("do_shuffle", bool, defaultValue=False) \
            .add_argument("do_predict", bool, defaultValue=False) \
            .add_argument("task_name", str) \
            .add_argument("tag_rules", dict) \
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

        cache = self.cache.group(self.max_scan_num)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

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

    def process_data(self,
                     batch_size: int,
                     eval_batch_size: int = None,
                     test_batch_size: int = None):
        if self.use_test:
            self.myData_test = FTDataSetV2(self.data_files[2],
                                           self.tokenizer,
                                           self.lexicon_tree,
                                           self.word_vocab,
                                           self.tag_vocab,
                                           self.max_word_num,
                                           self.max_seq_length,
                                           self.default_tag,
                                           self.tag_rules,
                                           do_predict=self.do_predict,
                                           cache=self.cache,
                                           expand=False)
            self.dataiter_test = DataLoader(self.myData_test,
                                            batch_size=test_batch_size)
        else:
            self.myData = FTDataSetV2(self.data_files[0],
                                      self.tokenizer,
                                      self.lexicon_tree,
                                      self.word_vocab,
                                      self.tag_vocab,
                                      self.max_word_num,
                                      self.max_seq_length,
                                      self.default_tag,
                                      self.tag_rules,
                                      do_shuffle=self.do_shuffle,
                                      cache=self.cache)

            self.dataiter = DataLoader(self.myData, batch_size=batch_size)
            if self.output_eval:
                self.myData_eval = FTDataSetV2(self.data_files[1],
                                               self.tokenizer,
                                               self.lexicon_tree,
                                               self.word_vocab,
                                               self.tag_vocab,
                                               self.max_word_num,
                                               self.max_seq_length,
                                               self.default_tag,
                                               self.tag_rules,
                                               cache=self.cache,
                                               expand=False)
                self.dataiter_eval = DataLoader(self.myData_eval,
                                                batch_size=eval_batch_size)

    def __call__(self):
        if self.use_test:
            return {
                'test_set': self.myData_test,
                'test_iter': self.dataiter_test,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }
        if self.output_eval:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'eval_set': self.myData_eval,
                'eval_iter': self.dataiter_eval,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }
        else:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }


class FTDataSetV2(Dataset):
    def __init__(
        self,
        file: str,
        tokenizer,
        lexicon_tree: Trie,
        word_vocab: Vocab,
        tag_vocab: Vocab,
        max_word_num: int,
        max_seq_length: int,
        default_tag: str,
        tag_rules: Dict[str, str],
        do_predict: bool = False,
        do_shuffle: bool = False,
        cache: FileCache = None,
        expand: bool = True,
    ) -> None:

        self.file: str = file
        self.tokenizer = tokenizer
        self.lexicon_tree: Trie = lexicon_tree
        self.word_vocab: Vocab = word_vocab
        self.label_vocab: Vocab = tag_vocab
        self.max_word_num: int = max_word_num
        self.max_seq_length: int = max_seq_length
        self.default_tag: str = default_tag
        self.tag_rules = tag_rules
        self.do_shuffle: bool = do_shuffle
        self.do_predict: bool = do_predict
        self.cache = cache
        self.expand = expand
        if not self.do_predict:
            self.init_dataset()

    def convert_prompt(self, obj):
        text, labels = obj["text"], obj["label"]
        entities = get_entities(labels, text)
        if len(entities)==0:
            encoding = self.tokenizer(''.join(text),max_length=self.max_seq_length,padding="max_length",truncation=True,return_tensors="pt")
            yield torch.zeros(self.max_seq_length).int(),encoding["input_ids"][0],encoding["attention_mask"][0],torch.zeros(self.max_seq_length).int(),int(0)
        for start, end, label, word in entities:

            def convert(start, end, label, word, positive=True):
                # if not positive:
                #     word.append("不")
                input_ids = tensor(
                    self.tokenizer.convert_tokens_to_ids(
                        ["[CLS]"] + text[:self.max_seq_length - 6 - len(word) -
                                         len(f"{self.tag_rules[label]}")] +
                        ["[SEP]"] + word +
                        list(f"是一个{self.tag_rules[label]}") +
                        ["[SEP]"])).int()
                attention_mask = torch.zeros(self.max_seq_length).int()
                attention_mask[:input_ids.shape[0]] = 1
                input_ids = torch.cat([
                    input_ids,
                    tensor([0] * (self.max_seq_length - input_ids.shape[0]))
                ])

                origin_attention_mask = torch.zeros(self.max_seq_length).int()
                origin_attention_mask[start + 1:end + 1] = 1
                entity_attention_mask = torch.zeros(self.max_seq_length).int()
                start = len(
                    text[:self.max_seq_length - 6 - len(word) -
                         len(f"{self.tag_rules[label]}")]) + 5 + len(word)
                end = start + len(self.tag_rules[label])
                entity_attention_mask[start:end] = 1
                return origin_attention_mask, input_ids, attention_mask, entity_attention_mask, int(
                    positive)

            yield convert(start, end, label, word)
            if not self.expand:
                break
            id = self.label_vocab.token2id(token=f"B-{label}")
            next_ids = (id + 1) % len(self.label_vocab)
            while self.label_vocab.id2token(next_ids)[0] != 'B':
                next_ids = (next_ids + 1) % len(self.label_vocab)
            assert next_ids != id
            yield convert(start, end,
                          self.label_vocab.id2token(next_ids)[2:], word, False)

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
        segment_ids = torch.ones(self.max_seq_length, dtype=torch.int)
        segment_ids[:len(token_ids)] = 0
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
            matched_word_ids[i][:len(word_ids)] = tensor(word_ids).int()
            matched_word_mask[i][:len(word_ids)] = 1

        if return_dict:
            return {
                "input_ids": input_token_ids,
                "token_type_ids": segment_ids,
                "attention_mask": attention_mask,
                "matched_word_ids": matched_word_ids,
                "matched_word_mask": matched_word_mask,
                "labels": labels,
            }

        return input_token_ids, segment_ids, attention_mask, matched_word_ids, matched_word_mask, labels

    def init_dataset(self):
        reader = FileReader(self.file)
        line_total = reader.line_size()
        self.input_token_ids = []
        self.input_entitiy_mask = []
        self.segment_ids = []
        self.attention_mask = []
        self.matched_word_ids = []
        self.matched_word_mask = []
        self.prompt_input_ids = []
        self.prompt_attention_mask = []
        self.prompt_entity_mask = []
        self.positive = []
        self.labels = []

        for line in tqdm(reader.line_iter(),
                         desc=f"load dataset from {self.file}",
                         total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            input_token_ids, segment_ids, attention_mask, matched_word_ids, matched_word_mask, labels = self.convert_embedding(
                data)
            for input_entity_mask, prompt_input_ids, prompt_attention_mask, prompt_entity_mask,  positive in self.convert_prompt(
                    data):
                self.prompt_input_ids.append(prompt_input_ids)
                self.prompt_attention_mask.append(prompt_attention_mask)
                self.input_token_ids.append(input_token_ids)
                self.input_entitiy_mask.append(input_entity_mask)
                self.segment_ids.append(segment_ids)
                self.attention_mask.append(attention_mask)
                self.matched_word_ids.append(matched_word_ids)
                self.matched_word_mask.append(matched_word_mask)
                self.prompt_entity_mask.append(prompt_entity_mask)
                self.positive.append(positive)
                self.labels.append(labels)

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
                torch.stack([self.segment_ids[i] for i in idx]),
                'matched_word_ids':
                torch.stack([self.matched_word_ids[i] for i in idx]),
                'matched_word_mask':
                torch.stack([self.matched_word_mask[i] for i in idx]),
                'prompt_input_ids':
                torch.stack([self.prompt_input_ids[i] for i in idx]),
                'prompt_attention_mask':
                torch.stack([self.prompt_attention_mask[i] for i in idx]),
                'positive': [self.positive[i] for i in idx],
                'input_entity_mask':[self.input_entitiy_mask[i] for i in idx],
                'prompt_entity_mask':[self.prompt_entity_mask[i] for i in idx],
                'labels':
                torch.stack([self.labels[i] for i in idx])
            }
        else:
            return {
                'input_ids': self.input_token_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'token_type_ids': self.segment_ids[idx],
                'matched_word_ids': self.matched_word_ids[idx],
                'matched_word_mask': self.matched_word_mask[idx],
                'prompt_input_ids': self.prompt_input_ids[idx],
                'positive': self.positive[idx],
                'prompt_attention_mask': self.prompt_attention_mask[idx],
                'input_entity_mask':self.input_entitiy_mask[idx],
                'prompt_entity_mask':self.prompt_entity_mask[idx],
                'labels': self.labels[idx]
            }

    def __len__(self):
        return self.size
