from CC.loaders.utils import *
from typing import *
import json
from tqdm import *


class VocabTag(Vocab):
    def __init__(self, default_tag="O"):
        super().__init__()
        self.token2tag: Dict = {}
        self.default_tag = default_tag

    def __add__(self, others):
        if isinstance(others, tuple):
            token, tag = others
            super().__add__(token)
            self.token2tag[token] = tag
        else:
            token = others
            super().__add__(token)
            self.token2tag[token] = [self.default_tag] * len(token)
        return self

    def tag(self, token: str):
        if isinstance(token, list):
            return [self.tag(t) for t in token]
        if token in self.token2tag:
            return self.token2tag[token]
        elif self.is_word:
            unk = f'<unk>{len(token)}'
            if unk in self.token2tag:
                return self.token2tag[unk]
            else:
                return self.token2tag['<unk>']
        else:
            print(f"token:{token} does not exist!")
            raise KeyError()

    def from_list(self, words: Iterable[Tuple[str, List[str]]], is_word: bool = False, has_default: bool = False, unk_num: str = 0):
        self.is_word = is_word
        if not has_default and self.is_word:
            self += "<pad>"
            self += "<unk>"
            for i in range(unk_num):
                self += f'<unk>{i+1}'
        for word in tqdm(words, desc="load vocab from list"):
            if isinstance(word, tuple):
                token, tag = word
                if not isinstance(tag, list):
                    raise ValueError(f"tag: {tag} must be a string list.")
                if isinstance(token, list):
                    ''.join(token)
                self += (token, tag)
            else:
                raise ValueError(f"word: {word} doesn't have the tag.")
        return self

    def from_files(self, files: List[str], is_word: bool = False, has_default: bool = False, unk_num: str = 0, max_scan_num: int = -1):
        words = []
        for file in files:
            file_lines = FileUtil.count_lines(file)
            if max_scan_num != -1:
                file_lines = min(file_lines, max_scan_num)
            for index, line in tqdm(enumerate(FileUtil.line_iter(file)), desc="load vocab from files", total=file_lines):
                if not line:
                    continue
                if index >= file_lines:
                    break
                data = json.loads(line)
                words.append((''.join(data[0]), data[1]))
        return self.from_list(words, is_word, has_default, unk_num)
