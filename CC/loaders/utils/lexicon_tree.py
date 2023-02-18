from collections import defaultdict
from typing import Dict, List


class TrieNode():
    """TrieNode
    """

    def __init__(self):
        self.children: Dict[TrieNode] = defaultdict(TrieNode)
        self.is_word = False


class Trie():
    """Trie
    """

    def __init__(self, use_single: bool = True):
        """init Trie

        Args:
            use_single (bool, optional): True set self.min_len = 0,False set self.min_len = 1. Defaults to True.
        """
        self.root: TrieNode = TrieNode()
        self.max_depth: int = 0
        if use_single:
            self.min_len: int = 0
        else:
            self.min_len: int = 1

    def insert(self, word: str):
        """insert word

        Args:
            word (str): the word 
        """
        ptr: TrieNode = self.root
        deep: int = 0
        for letter in word:
            ptr = ptr.children[letter]
            deep += 1
        ptr.is_word = True
        if deep > self.max_depth:
            self.max_depth = deep
        return self

    def __add__(self, word: str):
        """ add word

        Args:
            word (str): word

        Returns:
            Trie: self
        """
        return self.insert(word)

    def search(self, word: str) -> bool:
        """search word 

        Args:
            word (str): the word

        Returns:
            bool: True if exist, otherwise False
        """
        ptr: TrieNode = self.root
        for letter in word:
            ptr = ptr.children.get(letter)
            if ptr is None:
                return False
        return ptr.is_word

    def enumerateMatch(self, sent: str, space: str = "") -> List[str]:
        """enumerate words starting with sent[0]

        Args:
            sent (str): the sent
            space (str, optional): spilt string. Defaults to "".

        Returns:
            List[str]: the words starting with sent[0]. if there are more than one, remove the single word.
        """
        matched: List[str] = []
        ptr: TrieNode = self.root
        for i, letter in enumerate(sent):
            if i > self.max_depth:
                break
            ptr = ptr.children.get(letter)
            if ptr is None:
                break
            if i >= self.min_len and ptr.is_word:
                matched.append(space.join(sent[:i+1]))
        if len(matched) > 1 and len(matched[0]) == 1:
            matched = matched[1:]
        return matched

    def getAllMatchedWords(self, sent: str) -> List[str]:
        """Get All Matched Words

        Args:
            sent (str): to match sentence

        Returns:
            List[str]: list of words 
        """
        matched_set = set()
        for i in range(len(sent)):
            sub_sent = sent[i:]
            for word in self.enumerateMatch(sub_sent):
                matched_set.add(word)
        return sorted(matched_set)

    def getAllMatchedWordList(self, sent: str, max_words: None) -> List[List[str]]:
        matched: List[List[str]] = [[] for i in range(len(sent))]
        for i in range(len(sent)):
            sub_sent = sent[i:]
            words = self.enumerateMatch(sub_sent)
            if max_words is not None:
                words = words[:max_words]
            for word in words:
                for j in range(i+1, i+len(word)):
                    matched[j].append(word)
                if len(matched[i]) > 0 and len(word) == 1:
                    continue
                matched[i].append(word)
            if max_words is not None:
                matched[i] = matched[i][:max_words]
        return matched
