from . import *
import numpy as np
from tqdm import *
import os
import pickle


class WordEmbedding():
    def __init__(self):
        self.embedding_index = {}
        self.dimension = -1
        self.reader = None

    def build_from_txt(self,embedding_path: str,max_scan_num: int = 1000000,add_seg_vocab: bool = False):
        self.reader = FileReader(embedding_path)
        line_totals = self.reader.line_size()
        if add_seg_vocab or max_scan_num < 0:
            max_scan_num = line_totals
        else:
            max_scan_num = min(max_scan_num, line_totals)
        for index in tqdm(range(max_scan_num),desc="load word embedding...",
                                total=max_scan_num):
            line = self.reader.line(index)
            line = line.strip().split()
            if index == 0:
                self.dimension = int(line[1])
            elif len(line) == self.dimension+1:
                self.embedding_index[line[0]] = index
            elif len(line) > self.dimension+1:
                print(f"{line} length more than {self.dimension+1}")
                self.embedding_index[" ".join(line[:-self.dimension])] = index
            else:
                print(f"{line} embedding error")
        return self

    def get_embedding(self):
        return self.embedding_index, self.dimension, self.reader


class VocabEmbedding():
    def __init__(self, vocab: Vocab):
        self.vocab: Vocab = vocab
        self.dimension: int = 200

    def build_from_file(self,
                        embedding_path: str,
                        max_scan_num: int = 1000000,
                        add_seg_vocab: bool = False):
        embedding_index = {}
        if embedding_path is not None:
            embedding_index, self.dimension, embedding_reader = WordEmbedding() \
                .build_from_txt(embedding_path,max_scan_num=max_scan_num,add_seg_vocab=add_seg_vocab).get_embedding()
        self.embedding = np.empty([self.vocab.size, self.dimension])
        if embedding_reader is not None:
            for idx, word in tqdm(enumerate(self.vocab.idx2item),
                                  desc="load vocab embedding"):
                if word in embedding_index:
                    line = embedding_reader.line(embedding_index[word])
                    line = line.strip().split()
                    np_embedding = np.empty(self.dimension)
                    np_embedding = line[-self.dimension:]
                    self.embedding[idx, :] = np_embedding
                else:
                    self.embedding[idx, :] = self.random_embedding()
        return self

    def random_embedding(self) -> np.numarray:
        scale = np.sqrt(3.0 / self.dimension)
        return np.random.uniform(-scale, scale, (1, self.dimension))

    def get_embedding(self):
        return self.embedding, self.dimension
