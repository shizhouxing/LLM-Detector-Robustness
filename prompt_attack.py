# Copyright 2023 Zhouxing Shi, Yihan Wang, Fan Yin.
# Licensed under the BSD 3-Clause License.

import os
import json
import argparse
import numpy as np
import torch
from multiprocessing import Pool
from functools import partial
from dataset import load_data
from utils import read_text
from openai_adapters import get_chatgpt_response, get_chatgpt_response_content
from openai_adapters import openai_ai_text_detect, get_chatgpt_paraphrase
from nltk.corpus import stopwords


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='eli5', choices=['eli5', 'xsum'])
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--load', type=str, help='Load all the rounds')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--infer', action='store_true')

parser.add_argument('--num_train', type=int, default=50,
                    help='Number of examples for searching for the best prompt.')
parser.add_argument('--num_test', type=int, default=100,
                    help='Number of examples for testing.')
parser.add_argument('--num_iterations', type=int, default=10)
parser.add_argument('--num_candidates', type=int, default=10)
parser.add_argument('--num_prompts', type=int, default=1)
parser.add_argument('--num_repeat', type=int, default=1,
                    help='Number of repeated tests')
parser.add_argument('--max_matching_score', type=float, default=0.2)
parser.add_argument('--chatgpt_version', type=str, default='0301')
parser.add_argument('--num_processes', type=int, default=20)
parser.add_argument('--detector', type=str, default='openai', choices=['openai'])
parser.add_argument('--infer_method', type=str,
                    choices=['init', 'attack', 'human', 'paraphrase'])

parser.add_argument('--no_search_instruction', dest='search_instruction',
                    action='store_false',
                    help='Search for reference only but not instruction.')
parser.add_argument('--load_init', type=str, help='Load initial round')


args = parser.parse_args()


def write_text(path, text):
    with open(os.path.join(args.output_dir, path), 'w') as file:
        file.write(text)


class Prompt:
    instruction_default = 'Meanwhile please imitate the writing style and wording of the following passage:'

    def __init__(self, reference, instruction=None, results=None):
        self.reference = reference
        if instruction is None:
            instruction = Prompt.instruction_default
        self.instruction = instruction
        self.improved = False
        self.candidate_collected = False
        self.results = results

    def get_prompt(self):
        if self.reference:
            return f'{self.instruction}\n\n"{self.reference}"'
        else:
            return ''

    @property
    def score(self):
        return self.results['score'] if self.results else None

    @property
    def accuracy(self):
        return self.results['accuracy'] if self.results else None


class Candidate:
    def __init__(self, text, heuristic_score):
        self.text = text
        self.heuristic_score = heuristic_score

    def __lt__(self, other):
        return self.heuristic_score < other.heuristic_score


def get_post(question, prompt):
    if args.data == 'eli5':
        post = 'Please answer this question with at least 150 words:'
        post += f'\n\n{question}'
    elif args.data == 'xsum':
        post = 'Please complete this passage with at least 150 words:'
        post += f'\n\n"{question}"'
    else:
        raise NotImplementedError(args.data)
    if prompt != '':
        post += f'\n\n{prompt}'
    return post


def query_example(example, prompt='', suffix='', paraphrase=False,
                  init_output=None, human=False, **kwargs):
    i = example['index']
    example = example['example']

    if human:
        write_text(f'{i}_input{suffix}.txt', example['prompt'])
        output = example['human_text']
    elif paraphrase:
        output = get_chatgpt_paraphrase(init_output[i])
    else:
        post = get_post(example['prompt'], prompt)
        write_text(f'{i}_input{suffix}.txt', post)
        response = get_chatgpt_response(post, version=args.chatgpt_version, **kwargs)
        write_text(f'{i}_response{suffix}.json', json.dumps(response))
        output = get_chatgpt_response_content(response)

    write_text(f'{i}_output{suffix}.txt', output)
    if args.detector == 'openai':
        ret_detection = openai_ai_text_detect(output)
    else:
        raise NameError(args.detector)
    if args.verbose:
        print(f'Example {i}: detection {ret_detection}')
    write_text(f'{i}_detect{suffix}.json', json.dumps(ret_detection))

    ret = {
        'detection': ret_detection,
        'output': output,
        'score': ret_detection['score'],
    }
    return ret


def query(dataset, prompt='', suffix='', **kwargs):
    print(f'Querying: {len(dataset)} examples '
          f'with {args.num_processes} processes')

    examples = [{'example': example, 'index': i}
                for i, example in enumerate(dataset)]

    with Pool(args.num_processes) as pool:
        ret = pool.map(partial(query_example, suffix=suffix,
                                prompt=prompt, **kwargs), examples)

    count_correct = 0
    count_correct_long = 0
    avg_score = 0.
    for r in ret:
        correct = args.detector == 'openai' and r['detection']['score'] >= 90
        if correct:
            count_correct += 1
        if correct or len(r['output']):
            count_correct_long += 1
        avg_score += r['detection']['score']
    accuracy = count_correct * 100. / len(dataset)
    # When there is a requirement on the length
    accuracy_long = count_correct_long * 100. / len(dataset)
    avg_score = avg_score / len(dataset)
    print(f'Correct detections: {count_correct} ({accuracy:.2f}%)')
    print(f'Average score: {avg_score:.4f}')
    return {
        'examples': ret,
        'accuracy': accuracy,
        'accuracy_long': accuracy_long,
        'score': avg_score,
        'prompt': prompt,
        'suffix': suffix,
    }


def is_improved(ret_cand, prompt):
    if not prompt.reference:
        # If the reference is initially empty, we fill it regardless of
        # whether the attack is improved.
        return True
    else:
        return (ret_cand['accuracy'] < prompt.accuracy
                    or ret_cand['accuracy'] == prompt.accuracy
                    and ret_cand['score'] < prompt.score)


def search_reference(data, prompts, candidates, suffix):
    print('Searching for the reference')
    prompts_new = []
    results = []
    for i, prompt in enumerate(prompts):
        print(f'Prompt {i}: score {prompt.score:.4f} '
                f'acc {prompt.accuracy:.4f}')
        results.append([])

        if not prompt.candidate_collected:
            for example in prompt.results['examples']:
                candidates.append(Candidate(
                    example['output'], example['score']))
            prompt.candidate_collected = True
        if not candidates:
            break
        candidates.sort()

        for j in range(min(args.num_candidates, len(candidates))):
            cand = candidates.pop(0)
            score = cand.heuristic_score
            print(f'Candidate {j}: score {score:.4f}')
            prompt_new = Prompt(cand.text, prompt.instruction)
            prompt_new.results = ret_cand = query(
                data, prompt=prompt_new.get_prompt(),
                suffix=f'_{suffix}_candidate_{i}_{j}')
            results[-1].append(ret_cand)
            print('Score:', ret_cand['score'])
            print('Accuracy:', ret_cand['accuracy'])

            # Detect copying
            matching_score = get_matching_score(prompt_new.reference,
                                                ret_cand['examples'])
            print('Matching score:', matching_score)
            if matching_score >= args.max_matching_score:
                print('Exceeding the matching score threshold. '
                      'Discarding the prompt.')
                continue

            if is_improved(ret_cand, prompt):
                prompt.improved = True
                prompts_new.append(prompt_new)
                print('Improved')
            else:
                print('Not improved')

    return prompts_new, results


def get_matching_score(reference, examples):
    score = 0
    stop = set(stopwords.words('english'))
    words = [item for item in reference.split(' ') if item not in stop]
    for example in examples:
        for word in words:
            score += word in example['output']
    score /= len(examples) * len(words)
    return score


def search_instruction(data, prompts, suffix):
    print('Searching for the instruction')

    # From "LARGE LANGUAGE MODELS ARE HUMAN-LEVEL PROMPT ENGINEERS"
    prompt_variation = 'Generate a variation of the following instruction while keeping the semantic meaning:'

    prompts_new = []
    results = []
    for i, prompt in enumerate(prompts):
        print(f'Prompt {i}: score {prompt.score:.4f} '
                f'acc {prompt.accuracy:.4f}')
        results.append([])

        for j in range(args.num_candidates):
            post = f'{prompt_variation}\n\n"{prompt.instruction}"'
            write_text(f'{suffix}_prompt_variation_{i}_{j}_input.txt', post)
            response = get_chatgpt_response(post, version=args.chatgpt_version)
            output = get_chatgpt_response_content(response)
            write_text(f'{suffix}_prompt_variation_{i}_{j}_output.txt', output)
            # Sometimes the output contains quotation marks which should be removed
            if output.startswith('"') and output.endswith('"'):
                output = output[1:-1]

            prompt_new = Prompt(prompt.reference, output)
            prompt_new.results = ret_cand = query(
                data, prompt=prompt_new.get_prompt(),
                suffix=f'_{suffix}_candidate_{i}_{j}')
            results[-1].append(ret_cand)
            print('Score:', ret_cand['score'])
            print('Accuracy:', ret_cand['accuracy'])

            # Detect copying
            matching_score = get_matching_score(prompt_new.reference,
                                                ret_cand['examples'])
            print('Matching score:', matching_score)
            if matching_score >= args.max_matching_score:
                print('Exceeding the matching score threshold. '
                      'Discarding the prompt.')
                continue

            if is_improved(ret_cand, prompt):
                prompt.improved = True
                prompts_new.append(prompt_new)
                print('Improved')
            else:
                print('Not improved')

    return prompts_new, results


def search_prompt(data):
    print('Initial query')
    if args.load_init:
        ret_init = json.loads(read_text(
            os.path.join(args.load_init, 'ret_init.json')))
        assert len(ret_init['examples']) == len(data)
        print('Reuse loaded initial result')
        print(f'Accuracy {ret_init["accuracy"]}, score {ret_init["score"]}')
    else:
        ret_init = query(data, prompt='', suffix='_init')
    write_text('ret_init.json', json.dumps(ret_init))
    ret_all = [ret_init]
    prompts = [Prompt('', results=ret_init)]
    candidate_refs = []

    # Search for the attacking prompt
    for t in range(args.num_iterations):
        print(f'\nRound {t}:')

        # Iteratively search for reference and instruction
        if args.search_instruction:
            search_mode = 'reference' if t % 2 == 0 else 'instruction'
        else:
            search_mode = 'reference'

        if search_mode == 'reference':
            prompts_new, ret = search_reference(
                data, prompts, candidates=candidate_refs,
                suffix=f'round_{t}')
        elif search_mode == 'instruction':
            prompts_new, ret = search_instruction(
                data, prompts, suffix=f'round_{t}')
        else:
            raise NotImplementedError()
        ret_all.append(ret)

        prompts = ([prompt for prompt in prompts if not prompt.improved]
                        + prompts_new)
        prompts = sorted(prompts, key=lambda prompt:prompt.score)
        prompts = prompts[:args.num_prompts]
        write_text('results.json', json.dumps(ret_all))
        print('Score:', prompts[0].score)
        print('Accuracy:', prompts[0].accuracy)

    print('\n')
    print('Initial score:', ret_init['score'])
    print('Initial accuracy:', ret_init['accuracy'])
    print('Final score:', prompts[0].score)
    print('Final accuracy:', prompts[0].results['accuracy'])
    final_prompt = prompts[0].get_prompt()
    write_text('prompt.txt', final_prompt)
    return final_prompt


def inference(data):
    results = {
        'init': [], 'attack': [], 'paraphrase': [], 'human': [],
    }

    if args.infer_method in [None, 'init']:
        for i in range(args.num_repeat):
            print('Repeat', i)
            ret = query(data, suffix=f'_init_{i}')
            results['init'].append(ret)
        torch.save(results['init'], os.path.join(args.output_dir, 'init.pkl'))

    if args.infer_method in [None, 'human']:
        for i in range(args.num_repeat):
            print('Repeat', i)
            ret = query(data, suffix=f'_human_{i}', human=True)
            results['human'].append(ret)
        torch.save(results['human'], os.path.join(args.output_dir, 'human.pkl'))

    if args.infer_method in [None, 'attack']:
        path = os.path.join(args.load, 'prompt.txt')
        if not os.path.exists(path):
            # Legacy filename
            path = os.path.join(args.load, 'instruction.txt')
        prompt = read_text(path)
        results['attack'] = []
        for i in range(args.num_repeat):
            print('Repeat', i)
            ret = query(data, prompt=prompt, suffix=f'_attack_{i}')
            results['attack'].append(ret)
        torch.save(results['attack'], os.path.join(args.output_dir, 'attack.pkl'))

    if args.infer_method in [None, 'paraphrase']:
        if args.load_init:
            ret_init = torch.load(
                os.path.join(args.load_init, 'results.pkl'))['init']
        else:
            ret_init = results['init']

        for i in range(args.num_repeat):
            print('Repeat', i)
            ret = query(data, suffix=f'_paraphrase_{i}', paraphrase=True,
                        init_output=[r['output']
                                     for r in ret_init[i]['examples']])
            torch.save(ret, os.path.join(args.output_dir, 'attack.pkl'))
            results['paraphrase'].append(ret)
        torch.save(results['paraphrase'],
                   os.path.join(args.output_dir, 'paraphrase.pkl'))

    torch.save(results, os.path.join(args.output_dir, 'results.pkl'))

    print()
    print('Results')
    for k, ret in results.items():
        score = [r['score'] for r in ret]
        acc = [r['accuracy'] for r in ret]
        print(k)
        print('Score:', f'{np.mean(score):.2f}', f'{np.std(score):.4f}')
        print('Acc:', f'{np.mean(acc):.2f}', f'{np.std(acc):.4f}')
        print(f'{np.mean(score):.2f}±{np.std(score):.2f}')
        print(f'{np.mean(acc):.1f}±{np.std(acc):.2f}')
        print()


if __name__ == '__main__':
    data_train = load_data(
        args.data, num_examples=args.num_train, split='train')
    data_test = load_data(
        args.data, num_examples=args.num_test, split='test')
    print('Number of examples for the search:', len(data_train))
    print('Number of examples for testing:', len(data_test))
    print()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    torch.save(data_train, os.path.join(args.output_dir, 'data_train.pkl'))
    torch.save(data_test, os.path.join(args.output_dir, 'data_test.pkl'))

    if args.infer:
        inference(data_test)
    else:
        prompt = search_prompt(data_train)
