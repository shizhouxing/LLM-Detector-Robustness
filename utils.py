# Copyright 2023 Zhouxing Shi, Yihan Wang, Fan Yin.
# Licensed under the BSD 3-Clause License.

import json
import sys


def read_text(path):
    with open(path) as file:
        text = file.read()
    return text


def load_data(data_path, max_num=None):
    lines = []
    with open(data_path) as f:
        if max_num is not None:
            lines = [f.readline() for _ in range(max_num)]
        else:
            lines = list(f.readlines())

    def _cleanup(text):
        return text.replace('\n', ' ').strip()

    text = [_cleanup(json.loads(line)['text']) for line in lines]
    return text


def show_result(ret, fout=sys.stdout):
    text_before = ret.original_result.attacked_text
    text_after = ret.perturbed_result.attacked_text
    words_before = text_before._words
    words_after = text_after._words
    assert len(words_before) == len(words_after)
    fout.write(f'{len(words_before)} words\n')
    for before, after in zip(words_before, words_after):
        if before != after:
            fout.write(f'{before} -> {after}\n')


def save_results(ret, path='results/substitutions.txt'):
    with open(path, 'w') as fout:
        for i, item in enumerate(ret):
            fout.write(f'{i}\n')
            show_result(item, fout)
            fout.write('\n')


def stat_detection(ret):
    count_pos = 0
    count_total = 0
    count_invalid = 0
    for item in ret:
        if item['length'] < 1000:
            count_invalid += 1
            continue
        count_total += 1
        if item['pred_classname'] in ['likely', 'possibly']:
            count_pos += 1
    print('Invalid (too short):', count_invalid)
    print('Total valid:', count_total)
    print(f'Correct: {count_pos} '
          f'(percentage {count_pos * 1. / count_total * 100:.2f}%)')


def clip(text, clip_length):
    lines = text.split('\n')
    ret = ''
    for line in lines:
        if ret:
            ret += '\n'
        ret += line
        if len(ret) >= clip_length:
            break
    return ret
