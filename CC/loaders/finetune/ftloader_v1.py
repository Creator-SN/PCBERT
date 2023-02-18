from functools import lru_cache
import json
from typing import Any, Dict, List, Union
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.utils.dummy_pt_objects import BertModel
from CC.loaders.utils.cache_manager import FileCache
from CC.loaders.utils.embedding import VocabEmbedding
from CC.loaders.utils.lexicon_factory import TrieFactory
from CC.loaders.utils.lexicon_tree import Trie
from CC.loaders.utils.parser import KwargsParser
from CC.loaders.utils.reader import FileReader
from CC.loaders.utils.vocab import Vocab
from transformers import BertTokenizer, BertModel
from ICCSupervised.ICCSupervised import IDataLoader
import random
from torch import tensor
from tqdm import tqdm
import tempfile
import threading



class FTLoaderV1(IDataLoader):
    """Fine-Tune Loader Version 1
    feature:
        the label of matched word - sentence embedding
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
            .add_argument("tag_embedding_file",str) \
            .add_argument("bert_pretrain_path", str,
                description="Bert Pretrain Path, e.g: /model/bert/ , /model/bert contains config.json/vocab.txt/pytorch.bin") \
            .add_argument("external_entities_file",str) \
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
            .add_argument("lexicon_tree_cache_path", str, optional=True) \
            .add_argument("word_vacab_cache_path", str, optional=True) \
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

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_pretrain_path)
        self.encoder_model = BertModel.from_pretrained(self.bert_pretrain_path,output_hidden_states=True)
        self.encoder_model.eval()

        with open(self.external_entities_file, "r", encoding="utf-8") as f:
            self.external_entities = json.load(f)

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

        self.entity_tag_vocab: Vocab = Vocab().from_files(
            [self.tag_embedding_file], is_word=False, skip=1)

        # setup matched_word sentence embedding
        def label_load():
            word_label_embedding = {}
            word_label_embedding_dim = 200
            for word, idx in tqdm(self.word_vocab.item2idx.items(),
                                  desc="generate label embedding"):
                word_key = str(list(word))
                if word_key in self.external_entities["entities"]:
                    word_label_embedding[idx] = {}
                    for label, sentences in self.external_entities["entities"][
                            word_key]["labels"].items():
                        sentence = sentences[0]
                        encoding = self.tokenizer.encode_plus(
                            sentence["text"][:(
                                509 -
                                len(f"{word}是一个{self.tag_rules[label]}"))],
                            f"{word}是一个{self.tag_rules[label]}")
                        with torch.no_grad():
                            it = dict((k, torch.tensor(v).unsqueeze(0))
                                      for k, v in encoding.items())
                            output = self.encoder_model(**it)
                            embedding = output.hidden_states[-3][0][0]
                            word_label_embedding_dim = len(embedding)
                        word_label_embedding[idx][
                            self.entity_tag_vocab.token2id(
                                label)] = embedding.tolist()
            return word_label_embedding, word_label_embedding_dim

        self.word_label_embedding, self.word_label_embedding_dim = cache.load(
            "label_embedding_entities", lambda: label_load())

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
            self.myData_test = FTDataSetV1(
                self.data_files[2], self.tokenizer, self.lexicon_tree,
                self.word_vocab, self.tag_vocab, self.max_word_num,
                self.max_seq_length, self.default_tag, self.entity_tag_vocab,
                self.external_entities, self.max_label_num,
                self.word_label_embedding, self.word_label_embedding_dim,
                do_predict=self.do_predict,
                cache=self.cache)
            self.dataiter_test = DataLoader(self.myData_test,
                                            batch_size=test_batch_size)
        else:         
            self.myData = FTDataSetV1(self.data_files[0],
                                      self.tokenizer,
                                      self.lexicon_tree,
                                      self.word_vocab,
                                      self.tag_vocab,
                                      self.max_word_num,
                                      self.max_seq_length,
                                      self.default_tag,
                                      self.entity_tag_vocab,
                                      self.external_entities,
                                      self.max_label_num,
                                      self.word_label_embedding,
                                      self.word_label_embedding_dim,
                                      do_shuffle=self.do_shuffle,
                                      cache=self.cache)

            self.dataiter = DataLoader(self.myData, batch_size=batch_size)
            if self.output_eval:
                self.myData_eval =  FTDataSetV1(
                    self.data_files[1], self.tokenizer, self.lexicon_tree,
                    self.word_vocab, self.tag_vocab, self.max_word_num,
                    self.max_seq_length, self.default_tag,
                    self.entity_tag_vocab, self.external_entities,
                    self.max_label_num, self.word_label_embedding,
                    self.word_label_embedding_dim,
                    cache=self.cache)
                self.dataiter_eval = DataLoader(self.myData_eval,
                                                batch_size=eval_batch_size)

    def __call__(self):
        if self.use_test:
            return {
                'test_set': self.myData_test,
                'test_iter': self.dataiter_test,
                'vocab_embedding': self.vocab_embedding,
                'label_embedding': self.word_label_embedding,
                'embedding_dim': self.embedding_dim,
                'label_embedding_dim': self.word_label_embedding_dim,
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
                'label_embedding': self.word_label_embedding,
                'embedding_dim': self.embedding_dim,
                'label_embedding_dim': self.word_label_embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }
        else:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'vocab_embedding': self.vocab_embedding,
                'label_embedding': self.word_label_embedding,
                'embedding_dim': self.embedding_dim,
                'label_embedding_dim': self.word_label_embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }


class FTDataSetV1(Dataset):
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
        entity_tag_vocab,
        external_entities,
        max_label_num: int,
        word_label_embedding,
        word_label_embedding_dim: int = 768,
        do_predict: bool = False,
        do_shuffle: bool = False,
        cache:FileCache = None,
    ) -> None:

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
        self.external_entities = external_entities
        self.max_label_num = max_label_num
        self.word_label_embedding = word_label_embedding
        self.word_label_embedding_dim = word_label_embedding_dim
        self.cache = cache
        if not self.do_predict:
            self.init_dataset()

    def convert_embedding(self,
                          obj,
                          return_dict: bool = False):
        if "text" not in obj:
            raise ValueError("obj required attribute: text")
        text = ["[CLS]"] + obj["text"][:self.max_seq_length - 2] + ["[SEP]"]
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

        labels = torch.zeros(self.max_seq_length, dtype=torch.int)
        labels[:len(label_ids)] = tensor(label_ids[:self.max_seq_length]).int()
        # init input
        input_token_ids = torch.zeros(self.max_seq_length, dtype=torch.int)
        input_token_ids[:len(token_ids)] = tensor(token_ids[:self.max_seq_length]).int()
        segment_ids = torch.ones(self.max_seq_length, dtype=torch.int)
        segment_ids[:len(token_ids)] = 0
        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.int)
        attention_mask[:len(token_ids)] = 1
        matched_word_ids = torch.zeros((self.max_seq_length, self.max_word_num),
                                    dtype=torch.int)
        matched_word_mask = torch.zeros((self.max_seq_length, self.max_word_num),
                                     dtype=torch.int)
        matched_label_ids = torch.zeros(
            (self.max_seq_length, self.max_word_num, self.max_label_num),
            dtype=torch.int)
        matched_label_embeddings = torch.zeros(
            (self.max_seq_length, self.max_word_num, self.max_label_num,
             self.word_label_embedding_dim),
            dtype=torch.float)
        matched_label_mask = torch.zeros(
            (self.max_seq_length, self.max_word_num, self.max_label_num),dtype=torch.float)
        # get matched word
        matched_words = self.lexicon_tree.getAllMatchedWordList(
            text, self.max_word_num)
        for i, words in enumerate(matched_words):
            word_ids = self.word_vocab.token2id(words)
            matched_word_ids[i][:len(word_ids)] = tensor(word_ids).int()
            matched_word_mask[i][:len(word_ids)] = 1
            ids = []
            masks = [0] * self.max_label_num
            for word_index, word in enumerate(words):
                key = str(list(word))
                word_id = self.word_vocab.token2id(word)
                if key in self.external_entities["entities"]:
                    tags = list(self.external_entities["entities"][key]
                                ["labels"].keys())[:self.max_label_num]
                    tags = self.entity_tag_vocab.token2id(tags)
                    masks[:len(tags)] = [1] * len(tags)
                    matched_label_embeddings[i][word_index][:len(tags)] = tensor([
                        self.word_label_embedding[word_id][tag] for tag in tags
                    ]).float()
                    if len(tags) < self.max_label_num:
                        tags += [
                            self.entity_tag_vocab.token2id(self.default_tag)
                        ] * (self.max_label_num - len(tags))
                    ids.append(tags)
                else:
                    ids.append(
                        [self.entity_tag_vocab.token2id(self.default_tag)] *
                        self.max_label_num)
            if len(words) > 0:
                matched_label_ids[i][:len(ids)] = tensor(ids).int()
                matched_label_mask[i][:len(masks)] = tensor(masks).int()

        if return_dict:
            return {
                "input_ids": input_token_ids,
                "token_type_ids": segment_ids,
                "attention_mask": attention_mask,
                "matched_word_ids": matched_word_ids,
                "matched_word_mask": matched_word_mask,
                "matched_label_ids": matched_label_ids,
                "matched_label_mask": matched_label_mask,
                "matched_label_embeddings": matched_label_embeddings,
                "labels": labels,
            }

        return input_token_ids, segment_ids, attention_mask, matched_word_ids, matched_word_mask, matched_label_ids, matched_label_mask, matched_label_embeddings, labels

    def init_dataset(self):
        reader = FileReader(self.file)
        line_total = reader.line_size()
        self.input_token_ids = []
        self.segment_ids = []
        self.attention_mask = []
        self.matched_word_ids = []
        self.matched_word_mask = []
        self.matched_label_ids = []
        self.matched_label_mask = []
        # self.matched_label_embeddings = []
        self.matched_label_embeddings_path = tempfile.mkdtemp()
        self.labels = []
        cache = self.cache.group("temp_embedding")

        for index, line in tqdm(enumerate(reader.line_iter()),
                                desc=f"load dataset from {self.file}",
                                total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            input_token_ids, segment_ids, attention_mask, matched_word_ids, matched_word_mask, matched_label_ids, matched_label_mask, matched_label_embeddings, labels = self.convert_embedding(
                data)

            self.input_token_ids.append(input_token_ids)
            self.segment_ids.append(segment_ids)
            self.attention_mask.append(attention_mask)
            self.matched_word_ids.append(matched_word_ids)
            self.matched_word_mask.append(matched_word_mask)
            self.matched_label_ids.append(matched_label_ids)
            self.matched_label_mask.append(matched_label_mask)
            cache.save(f"{index}.pkl",matched_label_embeddings)
            self.labels.append(labels)

        self.size = len(self.input_token_ids)
        self.indexes = [i for i in range(self.size)]
        self.preload_thread:threading.Thread = threading.Thread(target=FTDataSetV1.__preloadding,args=(self,slice(0,1000,1)))
        self.preload_thread.start()
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __del__(self):
        if self.preload_thread.isAlive():
            self.preload_thread.join()

    def __preloadding(self,idx:Union[slice,int]):
        idx = self.indexes[idx]
        if isinstance(idx,list):
            for i in idx:
                self.__get_matched_label_embeddings(i)
        else:
            self.__get_matched_label_embeddings(idx)

    @lru_cache(2000)
    def __get_matched_label_embeddings(self,index):
        cache = self.cache.group("temp_embedding")
        return cache.load(f"{index}.pkl")

    def __getitem__(self, index):
        idx = self.indexes[index]
        matched_label_embeddings = []
        if isinstance(idx, list):
            for i in idx:
                matched_label_embeddings.append(self.__get_matched_label_embeddings(i))
            matched_label_embeddings = torch.stack(matched_label_embeddings)
            
            if not self.preload_thread.isAlive():
                self.preload_thread:threading.Thread = threading.Thread(target=FTDataSetV1.__preloadding,args=(self,slice(index.stop,index.stop+1000,1)))
                self.preload_thread.start()

            return {
                'input_ids': torch.stack([self.input_token_ids[i] for i in idx]),
                'attention_mask': torch.stack([self.attention_mask[i] for i in idx]),
                'token_type_ids': torch.stack([self.segment_ids[i] for i in idx]),
                'matched_word_ids': torch.stack([self.matched_word_ids[i] for i in idx]),
                'matched_word_mask': torch.stack([self.matched_word_mask[i] for i in idx]),
                'matched_label_ids': torch.stack([self.matched_label_ids[i] for i in idx]),
                'matched_label_mask': torch.stack([self.matched_label_mask[i] for i in idx]),
                'matched_label_embeddings': matched_label_embeddings,
                'labels': torch.stack([self.labels[i] for i in idx])
            }
        else:
            matched_label_embeddings = self.__get_matched_label_embeddings(idx)

            if not self.preload_thread.isAlive():
                self.preload_thread:threading.Thread = threading.Thread(target=FTDataSetV1.__preloadding,args=(self,slice(index+1,index+1001,1)))
                self.preload_thread.start()

            return {
                'input_ids': self.input_token_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'token_type_ids': self.segment_ids[idx],
                'matched_word_ids': self.matched_word_ids[idx],
                'matched_word_mask': self.matched_word_mask[idx],
                'matched_label_ids': self.matched_label_ids[idx],
                'matched_label_mask': self.matched_label_mask[idx],
                'matched_label_embeddings': matched_label_embeddings,
                'labels': self.labels[idx]
            }

    def __len__(self):
        return self.size
