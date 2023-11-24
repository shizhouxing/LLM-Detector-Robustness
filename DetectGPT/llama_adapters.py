"""Use Llama to generate perturbations."""
import json
import string
import math
import time
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import LlamaWrapper

punctuation_list = [t for t in string.punctuation]
stop_list = ["'s", "\u2014", "-", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
stops = set(stop_list + punctuation_list)

llama_model = None
llama_model_checkpoint_path = '/home/data/llama/hf_models/65B'


def llama_parse_response(path):
    mapping = {}
    with open(path) as f:
        responses = json.loads(f.read())
    for response in responses:
        word = response['word']
        line = response['replacement']

        line = line.split('\n')
        items = [a.strip()[2:].strip().strip('"').strip("'") for a in line]
        items = [a for a in items if len(a.split(' ')) < 4]
        mapping[word] = items[:-1][:5]

    return mapping
