# coding=utf-8

# Copyright 2023 Authors of "Red Teaming Language Model Detectors with Language Models"
# available at https://arxiv.org/abs/2305.19713

# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pprint import pprint
from functools import partial
import string
from dataset.load_data import load_data
import random
from datetime import datetime
from utils import generate_attack_with_lm_replacement, LlamaWrapper, fix_adv_example_with_lm, smart_tokenizer_and_embedding_resize
import json
from tqdm import tqdm

from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          T5Tokenizer,
                          T5ForConditionalGeneration,
                          LlamaForCausalLM,
                          LlamaTokenizer,
                          )

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=True,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-6.7b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=False,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=['eli5', 'xsum', 'wiki']
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--attack_method",
        type=str,
        choices=["GPT_replacement", "llama_replacement"]
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=20
    )

    parser.add_argument(
        "--num_subsentence",
        type=int,
        default=100
    )

    parser.add_argument(
        "--replacement_checkpoint_path",
        type=str,
    )

    parser.add_argument(
        "--replacement_tokenizer_path",
        type=str
    )

    parser.add_argument(
        "--num_replacement_retry",
        type=int,
        default=5
    )

    parser.add_argument(
        "--valid_factor",
        type=float,
        default=1.5
    )

    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom", "llama"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if "llama" in args.model_name_or_path:
            max_memory_mapping = {0:"0GB", 1:"45GB", 2:"45GB", 3:"45GB"}
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', load_in_8bit=True, max_memory=max_memory_mapping)
        else:
            model_class = LlamaForCausalLM if "llama" in args.model_name_or_path else AutoModelForCausalLM
            model = model_class.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if "llama" not in args.model_name_or_path:
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    tokenizer_class = LlamaTokenizer if "llama" in args.model_name_or_path else AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if "llama" in args.model_name_or_path:
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    return model, tokenizer, device

def generate_attack_with_T5(prompt, output_with_watermark, args):
    mask_model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large").to('cuda')
    mask_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large", model_max_length=512)

    tokenized = word_tokenize(output_with_watermark)
    num_token = len(tokenized)

    for i in range(int(num_token * args.test_ratio)):
        mask_index = random.randint(0, num_token - 1)
        masked_text = prompt + " " + " ".join(tokenized[:mask_index]) + '<extra_id_0>' + " ".join(tokenized[mask_index+1:])
        for p in string.punctuation:
            masked_text = masked_text.replace(" " + p, p)

        t5_tokens = mask_tokenizer(masked_text, return_tensors="pt").to("cuda")
        stop_id = mask_tokenizer.encode(f"<extra_id_1>")[0]
        output = mask_model.generate(**t5_tokens, output_scores = True, return_dict_in_generate=True, eos_token_id=stop_id, num_beams=50, num_return_sequences=20)

        decode_output = mask_tokenizer.batch_decode(output.sequences)

        def post_process(text):
            if '<extra_id_0>' in text:
                start = text.index('<extra_id_0>')
            else:
                start = -12

            if '<extra_id_1>' in text:
                end = text.index('<extra_id_1>')
            else:
                end = len(text)

            text = text[start + 12:end]
            return text

        decode_output = [post_process(output) for output in decode_output]

        for replace in decode_output:
            if replace.strip()!= tokenized[mask_index]:
                tokenized[mask_index] = replace

    replaced_text = " ".join(tokenized)
    for p in string.punctuation:
        replaced_text = replaced_text.replace(" " + p, p)

    return replaced_text, int(num_token * args.test_ratio)



def generate(prompt, args, model=None, device=None, tokenizer=None, max_new_tokens=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """

    print(f"Generating with {args}")

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens if max_new_tokens is None else max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True,
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    gen_kwargs.update(dict(output_scores=True, return_dict_in_generate=True))
    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)
    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately:
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark.sequences[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark.sequences[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark,
            decoded_output_with_watermark,
            args)

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction':
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence':
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float):
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else:
            lst_2d.append([format_names(k), f"{v}"])
    return lst_2d

def detect(input_text, args, device=None, tokenizer=None, return_green_token=False):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text, return_green_token=return_green_token)
        print(score_dict)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
        return output, args, score_dict["prediction"], score_dict["green_token"] if return_green_token else []
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
        return output, args, "", []


def main(args):
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    if args.attack_method == "llama_replacement":
        replacement_model = LlamaWrapper(args.replacement_checkpoint_path, args.replacement_tokenizer_path, device='cuda:1')
        watermark_processor = WatermarkLogitsProcessor(vocab=list(replacement_model.tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)
        replacement_model.generate = partial(
                                        replacement_model.generate,
                                        logits_processor=LogitsProcessorList([watermark_processor])
                                    )

    data = load_data(args.dataset, split='test', num_examples=100, seed=0)

    success = 0
    total = 0

    now = datetime.now()
    date_string=now.strftime("%m-%d-%Y-%H:%M:%S")
    output_path = f"./{date_string}-{args.dataset}-{args.attack_method}-{args.test_ratio}-{args.gamma}-{args.delta}.json"
    res = []
    # Generate and detect, report to stdout
    if not args.skip_model_load:
        for i, d in tqdm(enumerate(data)):
            input_text = d['prompt']
            human_text = d['human_text']
            term_width = 30
            print("#"*term_width)
            print("Prompt:")
            print(input_text)

            _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text,
                                                                                                args,
                                                                                                model=model,
                                                                                                device=device,
                                                                                                tokenizer=tokenizer)
            decoded_output_with_watermark = decoded_output_with_watermark.replace('\n', ' ')

            without_watermark_detection_result = detect(decoded_output_without_watermark,
                                                        args,
                                                        device=device,
                                                        tokenizer=tokenizer,
                                                        return_green_token=False)
            with_watermark_detection_result = detect(decoded_output_with_watermark,
                                                    args,
                                                    device=device,
                                                    tokenizer=tokenizer,
                                                    return_green_token=False)

            if args.attack_method == "GPT_replacement":
                attack_message, num_replacement = generate_attack_with_lm_replacement(input_text, decoded_output_with_watermark, args)
            elif args.attack_method == "llama_replacement":
                attack_message, num_replacement = generate_attack_with_lm_replacement(input_text, decoded_output_with_watermark, args, replacement_model)
            with_watermark_attack_detection_result = detect(attack_message,
                                                    args,
                                                    device=device,
                                                    tokenizer=tokenizer,
                                                    return_green_token=False)

            human_text = " ".join(human_text.split(" ")[:args.max_new_tokens])

            human_detection_result = detect(human_text,
                                            args,
                                            device=device,
                                            tokenizer=tokenizer)

            print("#"*term_width)
            print("Output without watermark:")
            print(decoded_output_without_watermark)
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            pprint(without_watermark_detection_result)
            print("-"*term_width)

            print("#"*term_width)
            print("Output with watermark:")
            print(decoded_output_with_watermark)
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            pprint(with_watermark_detection_result)
            print("-"*term_width)

            print("#"*term_width)
            print("Attack output with watermark:")
            print(attack_message)
            print("-"*term_width)
            print(f"Detection result @ {args.detection_z_threshold}:")
            pprint(with_watermark_attack_detection_result)
            print(f"Number of replacements: {num_replacement}")
            print("-"*term_width)

            res.append({
                "human_text": human_text,
                "human_zscore": human_detection_result[0][4][1],
                "watermark_output": decoded_output_with_watermark,
                "watermark_zscore": with_watermark_detection_result[0][4][1],
                "no_watermark_output": decoded_output_without_watermark,
                "no_watermark_zscore": without_watermark_detection_result[0][4][1],
                "attack_output": attack_message,
                "attack_zscore": with_watermark_attack_detection_result[0][4][1],
                "num_replacement": num_replacement
            })

            total += 1
            print(with_watermark_detection_result[-2])
            print(with_watermark_attack_detection_result[-2])
            if with_watermark_detection_result[-2] and not with_watermark_attack_detection_result[-2]:
                success += 1
            if total == args.num_examples:
                break

        print(success/total)

        json.dump(res, open(output_path, "w+"))

    return

if __name__ == "__main__":

    args = parse_args()
    args.test_ratio = args.test_ratio * args.valid_factor
    # Not all substitution candidates have a valid substitution. Only around 1/args.valid_factor words have valid substitutions when we use LLaMA-65B.
    print(args)

    main(args)
