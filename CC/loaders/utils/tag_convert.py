from typing import *


class TagConvert():
    def __init__(self, rules: Dict[str, str], default_tag: str = 'O', not_found_action: str = "exception"):
        self.rules: Dict[str, str] = rules
        self.default_tag: str = default_tag
        self.not_found_action: str = not_found_action

    def word2prompt(self, word: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        if isinstance(word, str):
            word = list(word)
        prompt_origin = word+list(f"是一个单词,")
        prompt = word + list(f"是一个单词,")
        # prompt = ['[MASK]' for _ in word] + list(
        #     f"是一个")+['[MASK]' for _ in self.rules[single_tag]]+[',']
        prompt_mask = [1] * len(prompt)
        prompt_tags = [self.default_tag] * len(prompt)
        assert len(prompt) == len(prompt_mask) and len(
            prompt_mask) == len(prompt_tags) and len(prompt) == len(prompt_origin)
        return prompt, prompt_mask, prompt_tags, prompt_origin

    def tag2prompt(self, tag: List[str], word: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        """ convert tag to a prompt

        Args:
            tag (List[str]): tag list. e.g. ['B-position','I-position']
            word (List[str]): character list. e.g. ['教','师']

        Raises:
            ValueError: the length of word is not equal to the tag
            KeyError: tag not found in rules

        Returns:
            List[str]: prompt. e.g. ['教','师','是','一','个','[MASK]','[MASK]',',']
            List[str]: prompt_mask e.g. ['1','1','1','1','1','0','0','1']
            List[str]: prompt_tags e.g. ['B-position','I-position','O','O','O','O','O','O']
        """
        if len(word) != len(tag):
            raise ValueError(f"the length of word is not equal to the tag")
        # tag[0] for example: B-position, get the tag: position
        if isinstance(word, str):
            word = list(word)
        single_tag = tag[0].split('-')[-1]
        if single_tag not in self.rules:
            if self.not_found_action == "exception":
                raise KeyError(f"tag: {single_tag} not found in rules")
            else:
                return tuple([None] * 4)
        prompt_origin = word+list(f"是一个{self.rules[single_tag]}")+[',']
        prompt = word + list(f"是一个") + \
            ['[MASK]' for _ in self.rules[single_tag]]+[',']
        # prompt = ['[MASK]' for _ in word] + list(
        #     f"是一个")+['[MASK]' for _ in self.rules[single_tag]]+[',']
        prompt_mask = [1 for _ in word] + \
            ([1]*3) + [0 for _ in self.rules[single_tag]] + [1]
        prompt_tags = tag + \
            [self.default_tag for _ in range(len(word), len(prompt))]
        assert len(prompt) == len(prompt_mask) and len(
            prompt_mask) == len(prompt_tags) and len(prompt) == len(prompt_origin)
        return prompt, prompt_mask, prompt_tags, prompt_origin
