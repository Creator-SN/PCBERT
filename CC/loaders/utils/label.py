from typing import Any, Dict, List, Union


middle_set = {"M", "I"}
outside_set = {"O"}
single_set = {"S"}
start_set = {"B"}
end_set = {"E"}


def get_entities(labels: List[str], text: List[str], return_dict: bool = False) -> Union[List[Any], Dict[str, Any]]:
    entity_collections: List[Any] = []
    word = []
    cur_label = None
    start = -1
    labels.append("B-")
    text.append("[SEP]")
    for i, (label, ch) in enumerate(zip(labels, text)):
        if label[0] in start_set or label[0] in outside_set or label[0] in single_set:
            if cur_label is not None:
                entity_collections.append((start, i, cur_label, word))
            word = []
            cur_label = None
            start = -1
        if label[0] in start_set or label[0] in single_set:
            cur_label = "-".join(label.split("-")[1:])
            start = i
        if label[0] in middle_set and cur_label is None:
            cur_label = "-".join(label.split("-")[1:])
            start = i
            # print(f"{'-'*10}error_entity{'-'*10}")
            # print(f"labels:{labels}\ntext:{text}")
        if label[0] not in outside_set:
            word.append(ch)
    labels.pop()
    text.pop()
    if not return_dict:
        return entity_collections
    return [{
            "start": start,
            "end": end,
            "label": label,
            "word": word
            } for start, end, label, word in entity_collections]


def get_labels(label: str, length: int, has_end=True, middle_symbol="I", has_single=True):
    labels = [f"{middle_symbol}-{label}"]*length
    if length == 1 and has_single:
        labels[0] = f"S-{label}"
        return labels
    if has_end:
        labels[-1] = f"E-{label}"
    labels[0] = f"B-{label}"
    return labels
