# Copyright 2023 Zhouxing Shi, Yihan Wang, Fan Yin.
# Licensed under the BSD 3-Clause License.

import datasets
import random


def load_data(data, split='test', num_examples=100, seed=0):
    if data == 'eli5':
        dataset = eli5(split=split, min_length=1000,
                       num_examples=num_examples, seed=0)
    elif data == 'xsum':
        dataset = xsum(split=split, min_length=1000, max_human_length=4096,
                       num_examples=num_examples, seed=seed)
    elif data == 'wiki':
        dataset = wiki(split=split, min_length=1000, num_examples=num_examples,
                       seed=seed)
    else:
        raise NameError(data)
    return dataset


def eli5(split, min_length=0, num_examples=None, seed=-1):
    if '[' not in split and not split.endswith('_eli5'):
        split = f'{split}_eli5'
    dataset = datasets.load_dataset('eli5', split=split)
    if seed != -1:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.filter(
        lambda example:
            max([len(text) for text in example['answers']['text']]) >= min_length)

    def _preprocess(example):
        longest_human_text = example['answers']['text'][0]
        for text in example['answers']['text'][1:]:
            if len(text) > len(longest_human_text):
                longest_human_text = text
        return {
            'prompt': example['title'],
            'human_text': longest_human_text,
        }

    res = []
    for example in dataset:
        res.append(_preprocess(example))
        if num_examples is not None and len(res) >= num_examples:
            break

    return res


def xsum(split, min_length=0, num_examples=None, max_human_length=4096, seed=-1):
    dataset = datasets.load_dataset('xsum', split=split)
    if seed != -1:
        dataset = dataset.shuffle(seed=seed)

    def _preprocess(example):
        sentences = example['document'].split('.')
        human_text = ''
        for sent in sentences[1:]:
            if human_text:
                human_text_ = human_text + '.' + sent
            else:
                human_text_ = sent
            if len(human_text_) > max_human_length:
                if human_text == '':
                    human_text = human_text_[:max_human_length]
                break
            else:
                human_text = human_text_
        human_text = human_text.strip()
        return {
            'prompt': sentences[0] + '.',
            'instruct_prompt': f"Write an article following the sentence: '{sentences[0] + '.'}'",
            'human_text': human_text,
        }
    dataset = dataset.map(_preprocess)

    if min_length > 0:
        dataset = dataset.filter(
            lambda example: len(example['human_text']) >= min_length)

    res = []
    for example in dataset:
        res.append(example)
        if num_examples is not None and len(res) >= num_examples:
            break

    return res


def wiki(split, min_length=0, num_examples=300, seed=-1):
    dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1', split=split)
    if seed != -1:
        dataset = dataset.shuffle(seed=seed)
        random.seed(seed)

    len_dataset = len(dataset)
    res = []

    while len(res) < num_examples:
        prompt_id = random.randint(0, len_dataset-1)
        prompt = []
        while prompt_id < len_dataset:
            if dataset[prompt_id]['text'] != "":
                prompt.append(dataset[prompt_id]['text'])
            prompt_id += 1
            if len(" ".join(prompt).split(" ")) >= 20:
                break

        human_answer = ''
        while prompt_id < len_dataset:
            if dataset[prompt_id]['text'] != "":
                if human_answer:
                    human_answer += ' '
                human_answer += dataset[prompt_id]['text']
            prompt_id += 1
            if len(human_answer) >= min_length:
                break

        if len(human_answer) >= min_length:
            res.append({
                'prompt': " ".join(prompt),
                'human_text': human_answer,
            })
    return res

res = load_data('wiki')
