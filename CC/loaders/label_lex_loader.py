from os import replace
from CC.loaders.utils import *
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import tensor
from transformers import BertTokenizer
from tqdm import *
from typing import *
from ICCSupervised.ICCSupervised import IDataLoader
import json
import numpy as np
import random


class LabelLXLoader(IDataLoader):
    def __init__(self, **args):
        KwargsParser(debug=True) \
            .add_argument("batch_size", int, defaultValue=4) \
            .add_argument("eval_batch_size", int, defaultValue=16) \
            .add_argument("test_batch_size", int, defaultValue=16) \
            .add_argument("word_embedding_file", str) \
            .add_argument("word_vocab_file", str) \
            .add_argument("word_vocab_file_with_tag", str) \
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
            .add_argument("default_tag", str, defaultValue="O") \
            .add_argument("use_test", bool, defaultValue=False) \
            .add_argument("do_shuffle", bool, defaultValue=False) \
            .add_argument("do_predict", bool, defaultValue=False) \
            .add_argument("lexicon_tree_cache_path", str, optional=True) \
            .add_argument("word_vacab_cache_path", str, optional=True) \
            .add_argument("task_name", str) \
            .add_argument("tag_rules", dict) \
            .add_argument("debug", bool, defaultValue=False) \
            .parse(self, **args)

        # get cache_key
        files = [self.train_file,self.eval_file,self.test_file,self.tag_file] 
        self.cache_key = [FileReader(file).etag() if file is not None else "None" for file in files]
        self.cache_key = "_".join(self.cache_key)
        self.cache = FileCache(f"./temp/{self.cache_key}")

        self.read_data_set()
        self.verify_data()
        self.process_data()

    def read_data_set(self):
        self.data_files = [self.train_file, self.eval_file, self.test_file]

        cache = self.cache.group(self.max_scan_num)

        # loading lexicon tree
        self.lexicon_tree = cache.load("lexicon_tree",lambda: TrieFactory.get_trie_from_vocabs(
            [self.word_vocab_file], self.max_scan_num))

        self.matched_words: List[str] = cache.load("matched_words",lambda: TrieFactory.get_all_matched_word_from_dataset(
            self.data_files, self.lexicon_tree))
        # restore all word_vocab_file_with_tag
        self.word_vocab = cache.load("word_vocab_tag",lambda: VocabTag().from_files(
            [self.word_vocab_file_with_tag], is_word=True, has_default=False, unk_num=5, max_scan_num=self.max_scan_num))

        matched_words_with_tags = [(word, self.word_vocab.tag(word))
                                   for word in self.matched_words]
        # re assign word_vocab for matched_words
        self.word_vocab = VocabTag().from_list(matched_words_with_tags,
                                               is_word=True, has_default=False, unk_num=5)

        self.tag_vocab = Vocab().from_files([self.tag_file],is_word=False)

        self.vocab_embedding, self.embedding_dim = cache.load("word_embedding",lambda: VocabEmbedding(self.word_vocab).build_from_file(
            self.word_embedding_file, self.max_scan_num, self.add_seq_vocab).get_embedding())

        self.tag_convert: TagConvert = TagConvert(self.tag_rules)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

    def verify_data(self):
        pass

    def process_data(self):
        if self.use_test:
            self.myData_test = LEXBertDataSet(self.data_files[2], **vars(self))
            self.dataiter_test = DataLoader(
                self.myData_test, batch_size=self.test_batch_size)
        else:
            self.myData = LEXBertDataSet(self.data_files[0], **vars(self))
            self.dataiter = DataLoader(self.myData, batch_size=self.batch_size)

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


class LEXBertDataSet(Dataset):
    def __init__(self, dataset_file: str, **args):
        self.file = dataset_file
        for name in args.keys():
            setattr(self, name, args[name])
        self.__init_dataset()

    def convert_embedding(self, item):
        if "text" not in item:
            raise KeyError(f"key text not exists in item: {item}")
        if not self.do_predict:
            if "label" not in item:
                raise KeyError(f"key label not exists in item: {item}")
            if "replace" not in item:
                raise KeyError(f"key replace not exists in item: {item}")
            # add guard
            item["text"].append("[SEP]")
            item["label"].append("B-Guard")

            origin_replace_entity = []
            for span in item["replace"]:
                origin_replace_entity.append((
                    span["start"], span["end"], span["origin"]))
            # resolve prompt
            prompts = []
            prompt_masks = []
            prompt_tags = []
            prompt_origins = []
            word = []
            labels = []
            exist_prompt = set()
            for ch, label in zip(item["text"], item["label"]):
                if label != self.default_tag:
                    if label.startswith('B-') or label.startswith("S-"):
                        if len(word) != 0:
                            prompt, prompt_mask, prompt_tag, prompt_origin = self.tag_convert.tag2prompt(
                                labels, word)
                            key = hash(str(prompt_origin))
                            if key not in exist_prompt:
                                exist_prompt.add(key)
                                prompts.append(prompt)
                                prompt_masks.append(prompt_mask)
                                prompt_tags.append(prompt_tag)
                                prompt_origins.append(prompt_origin)
                            word = []
                            labels = []
                    word.append(ch)
                    labels.append(label)
            # remove guard
            item["text"].pop()
            item["label"].pop()

            # resolve input
            text = ["[CLS]"] + item["text"][:self.max_seq_length-2]+["[SEP]"]
            origin_text = text[:]
            text_origin_length = len(text)

            matched_words = self.lexicon_tree.getAllMatchedWordList(
                text, self.max_word_num)
            # for matched_word add prompt
            for words in matched_words:
                for word in words:
                    tag = self.word_vocab.tag(word)
                    # if the word tag is "O", skip...
                    if tag[0] == self.default_tag:
                        continue
                    prompt, prompt_mask, prompt_tag, prompt_origin = self.tag_convert.tag2prompt(
                        tag, word)
                    key = hash(str(prompt_origin))
                    if key not in exist_prompt:
                        exist_prompt.add(key)
                        prompts.append(prompt)
                        prompt_masks.append(prompt_mask)
                        prompt_tags.append(prompt_tag)
                        prompt_origins.append(prompt_origin)

            label = [self.default_tag] + \
                item["label"][:self.max_seq_length-2]+[self.default_tag]
            mask = [1 for _ in text]
            for prompt, prompt_mask, prompt_tag, prompt_origin in zip(prompts, prompt_masks, prompt_tags, prompt_origins):
                if len(text)+len(prompt) <= self.max_seq_length:
                    text += prompt
                    label += prompt_tag
                    mask += prompt_mask
                    origin_text += prompt_origin

            # convert to ids
            token_ids = self.tokenizer.convert_tokens_to_ids(text)
            label_ids = self.tag_vocab.token2id(label)

            # replace origin labels
            for start, end, entity in origin_replace_entity:
                if end < self.max_seq_length:
                    origin_text[start+1:end+1] = entity

            origin_text = self.tokenizer.convert_tokens_to_ids(origin_text)

            labels = []
            for m, token_id in zip(mask, origin_text):
                if m == 0:
                    labels.append(token_id)
                else:
                    labels.append(-100)
            for start, end, _ in origin_replace_entity:
                if end < self.max_seq_length:
                    labels[start+1:end+1] = origin_text[start+1:end+1]

            np_input_ids = np.zeros(self.max_seq_length, dtype=np.int)
            np_input_ids[:len(token_ids)] = token_ids

            np_token_type_ids = np.ones(self.max_seq_length, dtype=np.int)
            np_token_type_ids[:text_origin_length] = 0

            np_attention_mask = np.ones(self.max_seq_length, dtype=np.int)
            np_attention_mask[:len(mask)] = mask

            np_label_ids = np.zeros(self.max_seq_length, dtype=np.int)
            np_label_ids[:len(label_ids)] = label_ids

            np_labels = np.zeros(self.max_seq_length, dtype=np.int)
            np_labels[:len(labels)] = labels
            np_labels[len(labels):] = -100

            np_origin_labels = np.zeros(self.max_seq_length, dtype=np.int)
            np_origin_labels[:len(origin_text)] = origin_text
            np_origin_labels[len(labels):] = -100

            assert np_input_ids.shape[0] == np_token_type_ids.shape[0]
            assert np_input_ids.shape[0] == np_attention_mask.shape[0]
            assert np_input_ids.shape[0] == np_label_ids.shape[0]
            assert np_input_ids.shape[0] == np_labels.shape[0]
            assert np_input_ids.shape[0] == np_origin_labels.shape[0]

            return np_input_ids, np_token_type_ids, np_attention_mask, np_labels, np_origin_labels, np_label_ids
        # TODO: predict

        raise NotImplemented("do_predict not implement")

    def __init_dataset(self):
        line_total = FileUtil.count_lines(self.file)
        self.input_token_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.origin_labels = []
        self.input_labels = []
        self.labels = []
        for line in tqdm(FileUtil.line_iter(self.file), desc=f"load dataset from {self.file}", total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            input_token_ids, token_type_ids, attention_mask, input_labels, origin_label, labels = self.convert_embedding(
                data)
            self.input_token_ids.append(input_token_ids)
            self.token_type_ids.append(token_type_ids)
            self.attention_mask.append(attention_mask)
            self.origin_labels.append(origin_label)
            self.input_labels.append(input_labels)
            self.labels.append(labels)
        self.size = len(self.input_token_ids)
        self.input_token_ids = np.array(self.input_token_ids)
        self.token_type_ids = np.array(self.token_type_ids)
        self.attention_mask = np.array(self.attention_mask)
        self.origin_labels = np.array(self.origin_labels)
        self.input_labels = np.array(self.input_labels)
        self.labels = np.array(self.labels)
        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        return {
            'input_ids': tensor(self.input_token_ids[idx]),
            'attention_mask': tensor(self.attention_mask[idx]),
            'token_type_ids': tensor(self.token_type_ids[idx]),
            'origin_labels': tensor(self.origin_labels[idx]),
            'input_labels': tensor(self.input_labels[idx]),
            'labels': tensor(self.labels[idx])
        }

    def __len__(self):
        return self.size
