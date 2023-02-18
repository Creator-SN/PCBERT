from CC.loaders.utils import *
from typing import *
from collections import defaultdict


class LabelSpan():
    def __init__(self) -> None:
        self.start: int = -1
        self.end: int = -1
        self.label: str = ""
        self.text: List[str] = []


class LabelCollection():
    def __init__(self, default_label: str = 'O') -> None:
        self.labels: Dict[str, List[List[str]]] = {}
        self.labels_set: Dict[str, Set[str]] = {}
        self.labels_length: Dict[str, Dict[int, List[str]]] = defaultdict(
            lambda: defaultdict(list))
        self.default_label = default_label

    def get_label_slice(self, text: List[str], labels: List[str]) -> List[LabelSpan]:
        items: List[LabelSpan] = []
        word: List[str] = []
        cur_label: str = None
        label_text_set: Set[str] = set()
        start: int = -1
        # set guard
        text.append("[SEP]")
        labels.append("B-guard")
        for index, (ch, label) in enumerate(zip(text, labels)):
            label = label.split("-")
            real_label = "-".join(label[1:])
            if label[0] == self.default_label or label[0] == "B" or label[0] == "S":
                if cur_label is not None:
                    key = hash(str(word)+cur_label)
                    if key not in label_text_set and start != -1 and len(word) > 0:
                        label_text_set.add(key)
                        span = LabelSpan()
                        span.label = cur_label
                        span.start = start
                        span.end = index
                        span.text = word
                        items.append(span)
                        start = -1
                    word = []
                if label[0] == "B" or label[0] == "S":
                    start = index
                    cur_label = real_label
                else:
                    start = -1
                    cur_label = None
            if label[0] == "M" and cur_label is None:
                start = index
                cur_label = real_label
            if label[0] != self.default_label:
                word.append(ch)
        # remove guard
        text.pop()
        labels.pop()
        return items

    def __add__(self, others) -> None:
        if isinstance(others, tuple):
            text, labels = others
            word = []
            cur_label = None
            # set guard
            text.append("[SEP]")
            labels.append("B-null")
            for ch, label in zip(text, labels):
                if label != self.default_label:
                    label = label.split('-')
                    if label[0] == 'B' or label[0] == "S" or label[0] == self.default_label:
                        label_str = '-'.join(label[1:])
                        if cur_label is not None:
                            if cur_label not in self.labels:
                                self.labels[cur_label] = []
                                self.labels_set[cur_label] = set()
                            key = "".join(word)
                            if key not in self.labels_set[cur_label] and len(word) > 0:
                                self.labels_set[cur_label].add(key)
                                self.labels[cur_label].append(word)
                                self.labels_length[cur_label][len(
                                    word)].append(word)
                            word = []
                        cur_label = label_str
                    if label[0] == "M" and cur_label is None:
                        cur_label = label_str
                    word.append(ch)
            text.pop()
            labels.pop()
        else:
            raise TypeError(f"element must be a tuple!")
        return self

    def __getitem__(self, label):
        if label not in self.labels:
            raise KeyError(f"label {label} not exists")
        return self.labels[label]
