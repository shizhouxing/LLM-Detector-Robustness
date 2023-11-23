# Copyright 2023 Zhouxing Shi, Yihan Wang, Fan Yin.
# Licensed under the BSD 3-Clause License.

import string

from nltk.tokenize import word_tokenize
from typing import Optional, Dict, Sequence

from transformers import (T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM)

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords = list(stopwords.words('english')) + ['also']


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: LlamaTokenizer,
    model: LlamaForCausalLM,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class LlamaWrapper:
    def __init__(self, checkpoint_path, tokenizer_path, device='cpu'):
        max_memory_mapping = {0:"45GB", 1:"45GB", 2:"45GB", 3:"45GB"}
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.model = LlamaForCausalLM.from_pretrained(checkpoint_path, device_map='auto', load_in_8bit=True, max_memory=max_memory_mapping)

        if self.tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token='[PAD]'),
                tokenizer=self.tokenizer,
                model=self.model,
            )

        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"

        self.tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

        self.device = device

    def to(self, device):
        self.model = self.model.to(device)

    def generate(self, prompts, **kwargs):
        tokenized_prompts = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.device)
        generate_output = self.model.generate(**tokenized_prompts, **kwargs)
        return generate_output


def choose_replacement(prompt, tokenized, mask_model, mask_tokenizer, test_ratio=0.1, start_position=0):
    candidates = []
    for i, token in enumerate(tokenized):
        if i == 0 or i == len(tokenized) - 1:
            continue
        if token in string.punctuation + "``''":
            continue
        if token[0].isupper() and i > 0 and tokenized[i-1] not in ['.', '!', '?', "'", '"']:
            continue
        if token[0] in ["'", "‘", '"', "”", '’', "“"]:
            continue
        if token.lower() in stopwords:
            continue
        if token[0].isdigit() or token in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero']:
            continue
        if token[0] in string.punctuation:
            continue
        masked_tokenized = [t for t in tokenized]
        masked_tokenized[i] = '<extra_id_0>'

        masked_text_pre = " ".join(masked_tokenized[:i])
        masked_text_after = " ".join(masked_tokenized[i+1:])
        for punctuation in ['.', ',', ':', ')', '"', "'", ';', '’', '”', '?', '!']:
            masked_text_pre = masked_text_pre.replace(" " + punctuation, punctuation)
            masked_text_after = masked_text_after.replace(" " + punctuation, punctuation)

        masked_text = prompt + " " + masked_text_pre + '<extra_id_0>' + masked_text_after

        input_ids = mask_tokenizer(masked_text, return_tensors="pt").input_ids.to("cuda")
        labels = mask_tokenizer(f"<extra_id_0>{token}<extra_id_1>", return_tensors="pt").input_ids.to("cuda")[:,:-2]


        outputs = mask_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        candidates.append((loss.cpu().detach().item(), token, i, mask_tokenizer.batch_decode(outputs.logits.argmax(-1))))

    num_candidate = int(len(tokenized) * test_ratio)
    if num_candidate == 0:
        candidate_list = []
    else:
        candidate_list = sorted(candidates)[-num_candidate:]

    candidate_list = [(a[2], a[1], a[3]) for a in candidate_list]
    return sorted(candidate_list)


def get_synonyms_llama(words, sentence, model):
    prompts = []
    for word in words:
        prompt = f'"{sentence}"\nSynonyms of the word "{word}" in the above sentence are:\na)'
        prompts.append(prompt)

    if len(prompts) == 0:
        return []
    decoded_outputs_res = []

    for i in range(len(prompts)):
        outputs = model.generate(prompts[i], max_new_tokens=10)
        decoded_outputs = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        decoded_outputs = [output.split('a)')[1] for output in decoded_outputs]
        decoded_outputs = [output.split('b)')[0] for output in decoded_outputs]
        decoded_outputs = [output.strip(' ').strip('\n').strip('"').strip("'").strip("’") for output in decoded_outputs]
        decoded_outputs = [output.split(',')[0].strip() for output in decoded_outputs]
        decoded_outputs_res.extend(decoded_outputs)

    replace_list = [(w, o) for (w, o, p) in zip(words, decoded_outputs_res, prompts) if len(o.split(' ')) <= 3 and w not in o and len(o) > 0 and o[0].isalpha()]
    print(replace_list)
    return replace_list


def fix_adv_example_with_lm(output, args, replacement_model=None):
    prompt = f"'{output}' with correct grammar is: \n"
    fixed_output = replacement_model.generate(prompt, max_new_tokens=len(prompt.split(' ')) + 5)


def generate_attack_with_lm_replacement(prompt, output_with_watermark, args, replacement_model=None):
    mask_model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large").to('cuda')
    mask_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large", model_max_length=512)

    subsentence = [prompt] + [a+'. ' for a in output_with_watermark.split('. ')]
    to_be_replaced = []

    curr_pos = 0
    for i in range(1, len(subsentence)):
        to_be_replaced_sub = choose_replacement(subsentence[i-1], word_tokenize(subsentence[i]), mask_model, mask_tokenizer, test_ratio=args.test_ratio)
        to_be_replaced_sub = [[a+curr_pos, b, c] for (a,b,c) in to_be_replaced_sub]
        curr_pos += len(word_tokenize(subsentence[i]))
        to_be_replaced.append(to_be_replaced_sub)

    for i in range(1, len(subsentence)):
        to_be_replaced_words = [a[1] for a in to_be_replaced[i-1]]
        best_sub_replacement = []
        for j in range(args.num_replacement_retry):
            if args.attack_method == 'GPT_replacement':
                sub_replacement = get_synonyms_chatGPT([a[1] for a in to_be_replaced[i-1]], subsentence[i-1]+subsentence[i])
            elif args.attack_method == 'llama_replacement':
                assert replacement_model is not None, "replacement model should be a llama model"
                sub_replacement = get_synonyms_llama([a[1] for a in to_be_replaced[i-1]], subsentence[i-1]+subsentence[i], replacement_model)
            else:
                raise NotImplementedError()

            sub_replacement = [a for a in sub_replacement if a[0] in to_be_replaced_words]
            print([a[0] for a in sub_replacement])
            print([a[1] for a in to_be_replaced[i-1]])

            if len(sub_replacement) > len(best_sub_replacement):
                best_sub_replacement = sub_replacement
            if len(sub_replacement) == len(to_be_replaced[i-1]):
                break
        for a in best_sub_replacement:
            if a[0] in to_be_replaced_words:
                index = to_be_replaced_words.index(a[0])
                to_be_replaced[i-1][index].append(a[1])
                to_be_replaced_words[index] = ""
    # [(a, b)]

    to_be_replaced = sum(to_be_replaced, [])
    to_be_replaced = [a for a in to_be_replaced if len(a) == 4]
    num_replacement = len(to_be_replaced)

    tokenized = sum([word_tokenize(a) for a in subsentence[1:]], [])

    for rep in to_be_replaced:
        tokenized[rep[0]] = rep[-1]


    text =  " ".join(tokenized)
    for punctuation in ['.', ',', ':', ')', '"', "'", ';', '’', '”', '?', '!']:
        text = text.replace(" " + punctuation, punctuation)

    text = text.replace("`` ", '"')
    text = text.replace("''", '"')
    text = text.replace("“ ", "“")
    text = text.replace("( ", "(")
    text = text.replace("’ ", "’")
    return text, num_replacement
