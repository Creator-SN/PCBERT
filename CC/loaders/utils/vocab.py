from typing import *
from CC.loaders.utils import *
from tqdm import *


class Vocab():

    def __init__(self):
        self.item2idx = {}
        self.idx2item = []
        self.size = 0

    def from_list(self, words: Iterable[str], is_word: bool = False, has_default: bool = False, unk_num: str = 0):
        """Load vocabs from array

        Args:
            words (Iterable[str]): word list
            is_word (bool, optional): is a word. Defaults to False.
            has_default (bool, optional): has default. Defaults to False.
            unk_num (str, optional): unk token number. Defaults to 0.

        Returns:
            Vocab: self
        """
        self.is_word = is_word
        if not has_default and self.is_word:
            self += "<pad>"
            self += "<unk>"
            for i in range(unk_num):
                self += f'<unk>{i+1}'
        for word in tqdm(words, desc="load vocab from list"):
            assert type(word) == str
            self += word
        return self

    def from_files(self, files: List[str], is_word: bool = False, has_default: bool = False, unk_num: str = 0, max_scan_num: int = -1,skip:int = 0):
        """get vocabs from file

        Args:
            files (List[str]): file name list
            is_word (bool, optional): is word. Defaults to False.
            has_default (bool, optional): has default value. Defaults to False.
            unk_num (str, optional): unkown number. Defaults to 0.

        Returns:
            Vocab: self
        """
        words = []
        for file in files:
            reader = FileReader(file)
            file_lines = reader.line_size()
            if max_scan_num != -1:
                file_lines = min(max_scan_num, file_lines)
            for index in tqdm(range(skip,file_lines), desc="load vocab from files"):
                line = reader.line(index)
                line = line.strip()
                if not line:
                    continue
                if index >= file_lines:
                    break
                word = line.split()[0].strip()
                words.append(word)
        return self.from_list(words, is_word, has_default, unk_num)

    def id2token(self, id: int):
        """ convert id to token

        Args:
            id (int): word index. if the type of id is list, convert the id list to the token list

        Returns:
            str: token or token list
        """
        if isinstance(id, list):
            return [self.id2token(index) for index in id]
        if id >= len(self.idx2item):
            raise ValueError("id out of range")
        return self.idx2item[id]

    def token2id(self, token: str):
        """ convert token to id

        Args:
            token (str): token or token list

        Raises:
            KeyError: if token does not exist

        Returns:
            int or List[int]: turen List[int] if token is List[str]. otherwise return int
        """
        if isinstance(token, list):
            return [self.token2id(t) for t in token]
        if token in self.item2idx:
            return self.item2idx[token]
        elif self.is_word:
            unk = f'<unk>{len(token)}'
            if unk in self.item2idx:
                return self.item2idx[unk]
            else:
                return self.item2idx['<unk>']
        else:
            print(f"token:{token} does not exist!")
            raise KeyError()

    def __add__(self, token: str):
        assert self.item2idx is not None
        assert self.idx2item is not None
        assert self.size is not None and type(self.size) == int
        self.item2idx[token] = self.size
        self.idx2item.append(token)
        self.size += 1
        return self

    def __len__(self):
        if self.idx2item is not None:
            return len(self.idx2item)
        raise ValueError("idx2item is None")
