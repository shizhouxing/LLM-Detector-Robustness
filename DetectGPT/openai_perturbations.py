"""Get perturbations by prompting ChatGPT."""

import json
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from openai_adapters import get_chatgpt_response

punctuation_list = [t for t in string.punctuation]
stop_list = ["'s", "\u2014", "-", "i", "me", "my", "myself", "we", "our", "ours",
             "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
             "him", "his", "himself", "she", "her", "hers", "herself", "it",
             "its", "itself", "they", "them", "their", "theirs", "themselves",
             "what", "which", "who", "whom", "this", "that", "these", "those",
             "am", "is", "are", "was", "were", "be", "been", "being", "have",
             "has", "had", "having", "do", "does", "did", "doing", "a", "an",
             "the", "and", "but", "if", "or", "because", "as", "until", "while",
             "of", "at", "by", "for", "with", "about", "against", "between",
             "into", "through", "during", "before", "after", "above", "below",
             "to", "from", "up", "down", "in", "out", "on", "off", "over",
             "under", "again", "further", "then", "once", "here", "there",
             "when", "where", "why", "how", "all", "any", "both", "each", "few",
             "more", "most", "other", "some", "such", "no", "nor", "not", "only",
             "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
             "just", "don", "should", "now"]
stops = set(stop_list + punctuation_list)


def get_perturbations(text, verbose=True):
    results = []
    sentences = sent_tokenize(text)
    def _call_lm(sent, wr):
        post = f"Given this sentence: '{sent}', for each word in {wr}, give 10 substitution words that do not change the meaning of the sentence. Return each word and its substitutions in one line, in the format of 'word:substitutions'\n\n"
        ret = get_chatgpt_response(post, verbose=verbose)
        print(ret)
        results.append(ret)

    for sent in sentences:
        words = word_tokenize(sent)
        sent_tag = nltk.pos_tag(words)
        to_be_replaced = [(i, t[0]) for (i, t) in enumerate(sent_tag)
                          if ((t[0] not in stops) and (t[0].lower() not in stops)
                              and ("www" not in t[0]))]
        if not len(to_be_replaced) == 0:
            to_be_replaced_index, to_be_replaced = tuple(zip(*to_be_replaced))
        wr = [f"'{w}'" for w in to_be_replaced]
        wr = ", ".join(wr)
        _call_lm(sent, wr)

    return results


def save_perturbations(results, idx, path=None):
    if path is None:
        path = f'results/chatgpt/{idx}.json'
    with open(path, 'w') as file:
        file.write(json.dumps(results))


def parse_mapping(line):
    line_ori = line
    line = line.strip()
    if not line:
        return
    if line.startswith('- '):
        line = line[2:].strip()
    if ':' in line:
        first_tokens = line.split(':')
    if not len(first_tokens) == 2:
        original = first_tokens[0].strip()
        substitution = []
    else:
        original = first_tokens[0].strip()
        substitution = first_tokens[1].split(',')
        substitution = [item.strip() for item in substitution]
    return original, substitution


def parse_response(path):
    mapping = {}
    with open(path) as f:
        responses = json.loads(f.read())
    for response in responses:
        assert len(response['choices']) == 1
        content = response['choices'][0]['message']['content']
        lines = content.split('\n')
        for line in lines:
            try:
                item = parse_mapping(line)
                if item is None:
                    continue
                mapping[item[0]] = item[1]
            except:
                print('==========Failed to parse=========')
                print(line)
                print('==================================')
    return mapping
