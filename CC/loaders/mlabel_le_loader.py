from CC.loaders.utils import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor
from transformers import BertTokenizer
from tqdm import *
from typing import *
from ICCSupervised.ICCSupervised import IDataLoader
import json
import numpy as np
import random


class MLabelLLoader(IDataLoader):
    def __init__(self, **args):
        KwargsParser(debug=True) \
            .add_argument("batch_size", int, defaultValue=4) \
            .add_argument("eval_batch_size", int, defaultValue=16) \
            .add_argument("test_batch_size", int, defaultValue=16) \
            .add_argument("word_embedding_file", str) \
            .add_argument("tag_embedding_file",str) \
            .add_argument("word_vocab_file", str) \
            .add_argument("external_entities_file",str) \
            .add_argument("train_file", str) \
            .add_argument("eval_file", str) \
            .add_argument("test_file", str) \
            .add_argument("tag_file", str) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("output_eval", bool, defaultValue=True) \
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
            .add_argument("ignore_rules",list,optional=True) \
            .parse(self, **args)

        # get cache_key
        files = [self.train_file,self.eval_file,self.test_file,self.tag_file] 
        self.cache_key = [FileReader(file).etag() if file is not None else "None" for file in files]
        self.cache_key = "_".join(self.cache_key)
        self.cache = FileCache(f"./temp/{self.cache_key}")

        self.read_data_set()
        self.verify_data()
        self.process_data(
            self.batch_size, self.eval_batch_size, self.test_batch_size)

    def read_data_set(self):      
        self.data_files: List[str] = [
            self.train_file, self.eval_file, self.test_file]
              
        # build lexicon tree
        cache = self.cache.group(self.max_scan_num)

        with open(self.external_entities_file,"r",encoding="utf-8") as f:
            self.external_entities = json.load(f)

        self.lexicon_tree = cache.load("lexicon_tree",lambda: TrieFactory.get_trie_from_vocabs(
                [self.word_vocab_file], self.max_scan_num))

        self.matched_words = cache.load("matched_words",lambda: TrieFactory.get_all_matched_word_from_dataset(
            self.data_files, self.lexicon_tree))

        # restore all word_vocab_file_with_tag
        self.word_vocab = cache.load("word_vocab",lambda: Vocab().from_list(
            self.matched_words, is_word=True, has_default=False, unk_num=5))

        self.tag_vocab: Vocab = Vocab().from_files(
            [self.tag_file], is_word=False)

        self.entity_tag_vocab:Vocab = Vocab().from_files([self.tag_embedding_file],is_word=False,skip=1)

        self.vocab_embedding,self.embedding_dim = cache.load("vocab_embedding",lambda: VocabEmbedding(self.word_vocab).build_from_file(
            self.word_embedding_file, self.max_scan_num, self.add_seq_vocab).get_embedding())

        self.entity_tag_embedding,self.entity_tag_embedding_dim = cache.load("tag_vocab_embedding",lambda: VocabEmbedding(self.entity_tag_vocab).build_from_file(
            self.tag_embedding_file, self.max_scan_num, self.add_seq_vocab).get_embedding())

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

    def verify_data(self):
        pass

    def process_data(self, batch_size: int, eval_batch_size: int = None, test_batch_size: int = None):
        if self.use_test:
            self.myData_test = LEBertDataSet(self.data_files[2], self.tokenizer, self.lexicon_tree,
                                            self.word_vocab, self.tag_vocab, self.max_word_num, self.max_seq_length, self.default_tag,self.entity_tag_vocab,self.external_entities,self.max_label_num, self.do_predict,ignore_rules=self.ignore_rules)
            self.dataiter_test = DataLoader(
                self.myData_test, batch_size=test_batch_size)
        else:
            self.myData = LEBertDataSet(self.data_files[0], self.tokenizer, self.lexicon_tree, self.word_vocab,
                                        self.tag_vocab, self.max_word_num, self.max_seq_length, self.default_tag,self.entity_tag_vocab,self.external_entities,self.max_label_num,do_shuffle=self.do_shuffle,ignore_rules=self.ignore_rules)

            self.dataiter = DataLoader(self.myData, batch_size=batch_size)
            if self.output_eval:
                self.myData_eval = LEBertDataSet(self.data_files[1], self.tokenizer, self.lexicon_tree, self.word_vocab,
                                                 self.tag_vocab, self.max_word_num,  self.max_seq_length, self.default_tag,self.entity_tag_vocab,self.external_entities,self.max_label_num, ignore_rules=self.ignore_rules)
                self.dataiter_eval = DataLoader(
                        self.myData_eval, batch_size=eval_batch_size)

    def __call__(self):
        if self.use_test:
            return {
                'test_set': self.myData_test,
                'test_iter': self.dataiter_test,
                'vocab_embedding': self.vocab_embedding,
                'label_embedding': self.entity_tag_embedding,
                'embedding_dim': self.embedding_dim,
                'label_embedding_dim': self.entity_tag_embedding_dim,
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
                'label_embedding': self.entity_tag_embedding,
                'embedding_dim': self.embedding_dim,
                'label_embedding_dim': self.entity_tag_embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }
        else:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'vocab_embedding': self.vocab_embedding,
                'label_embedding': self.entity_tag_embedding,
                'embedding_dim': self.embedding_dim,
                'label_embedding_dim': self.entity_tag_embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }


class LEBertDataSet(Dataset):
    def __init__(self, file: str, tokenizer, lexicon_tree: Trie, word_vocab: Vocab, tag_vocab: Vocab, max_word_num: int, max_seq_length: int, default_tag: str, entity_tag_vocab, external_entities, max_label_num:int,do_predict: bool = False, do_shuffle: bool = False,ignore_rules = None):
        self.file: str = file
        self.tokenizer = tokenizer
        self.lexicon_tree: Trie = lexicon_tree
        self.word_vocab: Vocab = word_vocab
        self.label_vocab: Vocab = tag_vocab
        self.entity_tag_vocab: Vocab = entity_tag_vocab
        self.max_word_num: int = max_word_num
        self.max_seq_length: int = max_seq_length
        self.default_tag: str = default_tag
        self.do_shuffle: bool = do_shuffle
        self.do_predict: bool = do_predict
        self.ignore_rules = ignore_rules
        self.external_entities = external_entities
        self.max_label_num = max_label_num
        if not self.do_predict:
            self.init_dataset()

    def convert_embedding(self, obj, return_dict: bool = False, to_tensor: bool = False):
        if "text" not in obj:
            raise ValueError("obj required attribute: text")
        text = ["[CLS]"] + obj["text"][:self.max_seq_length-2] + ["[SEP]"]
        if "label" not in obj and self.do_predict:
            label = [self.default_tag for i in range(self.max_seq_length)]
        elif "label" not in obj:
            raise ValueError("obj required attribute: label")
        else:
            label = [self.default_tag] + \
                obj["label"][:self.max_seq_length-2]+[self.default_tag]
        # convert to embedding
        token_ids = self.tokenizer.convert_tokens_to_ids(text)
        label_ids = self.label_vocab.token2id(label)

        labels = np.zeros(self.max_seq_length, dtype=np.int)
        labels[:len(label_ids)] = label_ids[:self.max_seq_length]
        # init input
        input_token_ids = np.zeros(self.max_seq_length, dtype=np.int)
        input_token_ids[:len(token_ids)] = token_ids[:self.max_seq_length]
        segment_ids = np.ones(self.max_seq_length, dtype=np.int)
        segment_ids[:len(token_ids)] = 0
        attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
        attention_mask[:len(token_ids)] = 1
        matched_word_ids = np.zeros(
            (self.max_seq_length, self.max_word_num), dtype=np.int)
        matched_word_mask = np.zeros(
            (self.max_seq_length, self.max_word_num), dtype=np.int)
        matched_label_ids = np.zeros((self.max_seq_length,self.max_word_num,self.max_label_num),dtype=np.int)
        # get matched word
        matched_words = self.lexicon_tree.getAllMatchedWordList(
            text, self.max_word_num)
        for i, words in enumerate(matched_words):
            word_ids = self.word_vocab.token2id(words)
            matched_word_ids[i][:len(word_ids)] = word_ids
            matched_word_mask[i][:len(word_ids)] = 1
            ids = []
            for word in words:
                key = str(list(word))
                if key in self.external_entities["entities"]:
                    tags = self.external_entities["entities"][key]["label"][:self.max_label_num]
                    try:
                        tags = self.entity_tag_vocab.token2id(tags)
                    except :
                        print(tags)
                        raise
                    if len(tags)<self.max_label_num:
                        tags+=[self.entity_tag_vocab.token2id(self.default_tag)] * (self.max_label_num-len(tags))
                    ids.append(tags)
                else:
                    ids.append([self.entity_tag_vocab.token2id(self.default_tag)] * self.max_label_num)
            if len(words)>0:
                matched_label_ids[i][:len(ids)] = ids

        if to_tensor:
            input_token_ids = tensor(input_token_ids)
            segment_ids = tensor(segment_ids)
            attention_mask = tensor(segment_ids)
            matched_word_ids = tensor(matched_word_ids)
            matched_word_mask = tensor(matched_word_mask)
            labels = tensor(labels)
        if return_dict:
            return {
                "input_ids": input_token_ids,
                "token_type_ids": segment_ids,
                "attention_mask": attention_mask,
                "matched_word_ids": matched_word_ids,
                "matched_word_mask": matched_word_mask,
                "matched_label_ids": matched_label_ids,
                "labels": labels,
            }

        return input_token_ids, segment_ids, attention_mask, matched_word_ids, matched_word_mask,matched_label_ids, labels

    def init_dataset(self):
        line_total = FileUtil.count_lines(self.file)
        self.input_token_ids = []
        self.segment_ids = []
        self.attention_mask = []
        self.matched_word_ids = []
        self.matched_word_mask = []
        self.matched_label_ids = []
        self.labels = []

        for line in tqdm(FileUtil.line_iter(self.file), desc=f"load dataset from {self.file}", total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            input_token_ids, segment_ids, attention_mask, matched_word_ids, matched_word_mask,matched_label_ids, labels = self.convert_embedding(
                data)

            self.input_token_ids.append(input_token_ids)
            self.segment_ids.append(segment_ids)
            self.attention_mask.append(attention_mask)
            self.matched_word_ids.append(matched_word_ids)
            self.matched_word_mask.append(matched_word_mask)
            self.matched_label_ids.append(matched_label_ids)
            self.labels.append(labels)

        self.size = len(self.input_token_ids)
        self.input_token_ids = np.array(self.input_token_ids)
        self.segment_ids = np.array(self.segment_ids)
        self.attention_mask = np.array(self.attention_mask)
        self.matched_word_ids = np.array(self.matched_word_ids)
        self.matched_word_mask = np.array(self.matched_word_mask)
        self.matched_label_ids = np.array(self.matched_label_ids)
        self.labels = np.array(self.labels)
        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        return {
            'input_ids': tensor(self.input_token_ids[idx]),
            'attention_mask': tensor(self.attention_mask[idx]),
            'token_type_ids': tensor(self.segment_ids[idx]),
            'matched_word_ids': tensor(self.matched_word_ids[idx]),
            'matched_word_mask': tensor(self.matched_word_mask[idx]),
            'matched_label_ids': tensor(self.matched_label_ids[idx]),
            'labels': tensor(self.labels[idx])
        }

    def __len__(self):
        return self.size