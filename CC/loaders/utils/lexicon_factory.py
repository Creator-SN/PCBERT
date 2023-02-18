from typing import List
from . import *
from tqdm import *
import json


class TrieFactory():

    @staticmethod
    def get_trie_from_vocabs(vocab_files: List[str], max_line: int = -1) -> Trie:
        """build lexicon trie from vocab file

        Args:
            vocab_files (List[str]): list of files
            max_line (int, optional): maximum line. Defaults to None.

        Returns:
            Trie: builded trie
        """
        vocabs = set()
        for file in vocab_files:
            reader = FileReader(file)
            file_lines = reader.line_size()
            if max_line != -1:
                file_lines = min(max_line, file_lines)
            for index in tqdm(range(file_lines), desc="load vocabs into trie", total=file_lines):
                line = reader.line(index)
                line = line.strip().split()
                word = line[0].strip()
                vocabs.add(word)
        lexicon_tree: Trie = Trie()
        for word in tqdm(vocabs, desc="build trie"):
            lexicon_tree.insert(word)
        return lexicon_tree

    @staticmethod
    def get_all_matched_word_from_dataset(dataset_files: List[str], lexicon_tree: Trie) -> List[str]:
        """Get All Matched word from dataset json file

        Args:
            dataset_files (List[str]): dataset json file path, a json object each line
            lexicon_tree (Trie): lexicon trie

        Returns:
            List[str]: all matched words are sorted in order
        """
        matched_words = set()
        for file in dataset_files:
            reader = FileReader(file)
            file_lines = reader.line_size()
            for line in tqdm(reader.line_iter(), desc="load dataset matched word", total=file_lines):
                data = json.loads(line.strip())
                assert 'text' in data, 'dataset type error, expected text property in object'
                text = data['text']
                sent = [ch for ch in text]
                for word in lexicon_tree.getAllMatchedWords(sent):
                    matched_words.add(word)
        return sorted(matched_words)
