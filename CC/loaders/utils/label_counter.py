from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Set
from text2vec import SBert, cos_sim
from functools import lru_cache


class LabelCounter():

    def __init__(self) -> None:
        self.sim_model = SBert('paraphrase-mpnet-base-v2')
        self.label_counter = Counter()
        self.label_notrepeat_counter = Counter()
        # collections
        self.label_repeat: Dict[str, Dict[int, List[List[str]]]] = defaultdict(lambda:
            defaultdict(list))
        self.label_notrepeat: Dict[str, Dict[int, List[List[str]]]] = defaultdict(lambda:
            defaultdict(list))
        self.label_keys: Set = set()
        self.begin_set = {"B"}
        self.middle_set = {"I", "M"}
        self.end_set = {"E"}
        self.single_set = {"S"}
        self.outside_set = {"O"}

    def add(self, labels: List[str], text: List[str]) -> LabelCounter:
        # add guard
        labels.append("B-guard")
        text.append("[SEP]")
        cur_label = ""
        word = []
        for label, ch in zip(labels, text):
            if label[0] in self.begin_set or label[0] in self.single_set or label[0] in self.outside_set:
                if len(word) > 0:
                    if cur_label != "":
                        # add entity
                        self.label_counter[cur_label] += 1
                        self.label_repeat[cur_label][len(word)].append(word[:])
                        key = hash(f"{cur_label}-{word}")
                        if key not in self.label_keys:
                            self.label_keys.add(key)
                            self.label_notrepeat_counter[cur_label] += 1
                            self.label_notrepeat[cur_label][len(
                                word)].append(word[:])
                cur_label = "-".join(label.split("-")[1:])
                word = []
            if label[0] in self.middle_set and len(word) == 0:
                print(f"{'-'*10}error entity{'-'*10}")
                cur_label = "-".join(label.split("-")[1:])
            if label[0] not in self.outside_set:
                word.append(ch)
        labels.pop()
        text.pop()
        return self

    def pick(self, label: str, origin_word: List[str], k: int = 1, p: float = 0.6) -> List[List[str]]:
        label = label.split("-")
        if label[0] in self.begin_set or label[0] in self.middle_set or label[0] in self.end_set or label[0] in self.single_set:
            label = "-".join(label[1:])
        else:
            label = '-'.join(label)
        candidates = self.label_notrepeat[label][len(origin_word)]
        origin_word_str = ''.join(origin_word)
        sim_list = sorted(
            [(self.word_sim(*sorted([''.join(word), origin_word_str])), word) for word in candidates if origin_word_str != ''.join(word)], reverse=True)
        print(sim_list)
        pick_word = [word[:] for sim, word in sim_list if sim > p][:k]
        return pick_word

    @ lru_cache(maxsize=1000000)
    def word_sim(self, word1: str, word2: str) -> float:
        # use LRU cache
        a = self.sim_model.encode(word1)
        b = self.sim_model.encode(word2)
        return cos_sim(a, b)[0][0]
