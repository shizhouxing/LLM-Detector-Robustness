import time
import torch
import nltk
import copy
import string
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.max_memory_mapping = {0: "45GB", 1: "45GB", 2: "45GB"}
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.device = torch.device("cuda:1")
        self.model.to(self.device)
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.to(self.device) for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return prefix


class genetic_attack_agent():
    def __init__(self, tokenizer, model, device):
        self.language = 'English'
        self.pop_size = 10
        self.max_iter_rate = 0.2
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        super().__init__()

    def op_sent(self, sent_lst1, op, pos):
        sent_lst1_at = sent_lst1[:pos] + [op] + sent_lst1[pos + 1:]
        return sent_lst1_at

    def get_ll(self, text):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)

            labels = tokenized.input_ids
            return -self.model(**tokenized, labels=labels).loss.item()

    def select_best_replacement(self, idx, sent_cur, replace_list, sent_lst1):
        new_x_list = [self.op_sent(sent_cur, w, idx) for w in replace_list]
        logit_lst = []
        for sent_p in new_x_list:
            logit = self.get_ll(' '.join(sent_p))
            logit_lst.append(logit)
        indices = np.argsort(np.array(logit_lst), axis=0)
        return new_x_list[indices[0]]

    def generate_population(self, neighbours_lst, w_select_probs, example):
        return [self.perturb(example, neighbours_lst, w_select_probs, example) for _ in range(self.pop_size)]

    def crossover(self, sent1, sent2):
        sent_new = copy.deepcopy(sent1)
        for i in range(len(sent_new)):
            if np.random.uniform() < 0.5:
                sent_new[i] = sent2[i]
        return sent_new

    def perturb(self, sent_cur, neighbours_list, w_select_probs, sent_lst1):
        x_len = w_select_probs.shape[0]
        random_idx = np.random.choice(x_len, size=1, p=w_select_probs)[0]

        while sent_cur[random_idx] != sent_lst1[random_idx] and np.sum(sent_lst1 != sent_cur) < np.sum(
                np.sign(w_select_probs)):
            random_idx = np.random.choice(x_len, size=1, p=w_select_probs)[0]
        replace_list = neighbours_list[random_idx]
        return self.select_best_replacement(random_idx, sent_cur, replace_list, sent_lst1)

    def random_perturb(self, sent_cur, neighbours_list, w_select_probs):
        x_len = w_select_probs.shape[0]
        max_iter = 0
        for i in w_select_probs:
            if not i == 0:
                max_iter += 1
        max_perturbs = int(max_iter * self.max_iter_rate)
        random_idx = np.random.choice(x_len, size=max_perturbs, p=w_select_probs)
        for i in random_idx:
            replace_list = neighbours_list[i]
            w_idx = np.random.choice(len(neighbours_list[i]), size=1)
            w = replace_list[w_idx[0]]
            sent_cur = self.op_sent(sent_cur, w, i)
        return sent_cur

    def random_replacement(self, eval_example, mapping):
        tokenized = word_tokenize(eval_example)
        sent_tag = nltk.pos_tag(tokenized)

        to_be_replaced = [(i, t[0]) for (i, t) in enumerate(sent_tag)]
        to_be_replaced_index, to_be_replaced = tuple(zip(*to_be_replaced))

        neighbours_lst = []
        for i, word in enumerate(tokenized):
            if i in to_be_replaced_index:
                if word.capitalize() in mapping:
                    neighbours_lst.append(mapping[word.capitalize()])
                elif word in mapping:
                    neighbours_lst.append(mapping[word])
                elif "'" + word + "'" in mapping:
                    neighbours_lst.append(mapping["'" + word + "'"])
                else:
                    neighbours_lst.append([])
            else:
                neighbours_lst.append([])

        neighbours_len = [len(i) for i in neighbours_lst]
        w_select_probs = neighbours_len / np.sum(neighbours_len)
        pop = self.random_perturb(word_tokenize(eval_example), neighbours_lst, w_select_probs)
        return [pop]

    def attack(self, eval_example, mapping):
        tokenized = word_tokenize(eval_example)
        sent_tag = nltk.pos_tag(tokenized)
        to_be_replaced = [(i, t[0]) for (i, t) in enumerate(sent_tag)]
        to_be_replaced_index, to_be_replaced = tuple(zip(*to_be_replaced))

        neighbours_lst = []
        for i, word in enumerate(tokenized):
            if i in to_be_replaced_index:
                if word.capitalize() in mapping:
                    neighbours_lst.append(mapping[word.capitalize()])
                elif word in mapping:
                    neighbours_lst.append(mapping[word])
                elif "'" + word + "'" in mapping:
                    neighbours_lst.append(mapping["'" + word + "'"])
                else:
                    neighbours_lst.append([])
            else:
                neighbours_lst.append([])
        neighbours_len = [len(i) for i in neighbours_lst]
        w_select_probs = neighbours_len / np.sum(neighbours_len)
        pop = self.generate_population(neighbours_lst, w_select_probs, word_tokenize(eval_example))
        max_iter = 0
        for i in w_select_probs:
            if not i == 0:
                max_iter += 1
        flag = 0
        max_iter = int(max_iter * self.max_iter_rate)
        attacked_sent = None
        for i in range(max_iter):
            logit_lst = []
            for sent_p in pop:
                joined_sent = ' '.join(sent_p)
                for punctuation in string.punctuation:
                    joined_sent = joined_sent.replace(" " + punctuation, punctuation)
                logit = self.get_ll(joined_sent)
                logit_lst.append(logit)
            if flag:
                break
            indices = np.argsort(np.array(logit_lst), axis=0)
            elite = [pop[indices[0]]]
            select_probs = np.array(logit_lst)
            select_probs /= select_probs.sum()
            p1 = np.random.choice(self.pop_size, size=self.pop_size - 1, p=select_probs)
            p2 = np.random.choice(self.pop_size, size=self.pop_size - 1, p=select_probs)
            childs = [self.crossover(pop[p1[idx]], pop[p2[idx]]) for idx in range(self.pop_size - 1)]
            childs = [self.perturb(x, neighbours_lst, w_select_probs, word_tokenize(eval_example)) for x in childs]
            pop = elite + childs
        return elite