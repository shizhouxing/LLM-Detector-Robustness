# Copyright 2023 Zhouxing Shi, Yihan Wang, Fan Yin.
# Licensed under the BSD 3-Clause License.

"""Use ChatGPT to generate perturbations."""
import string
import openai
import math
import time
import os


punctuation_list = [t for t in string.punctuation]
with open('openai_key.txt') as file:
    openai_key = file.read().strip()
openai.api_key = openai_key
print('Setting openai.api_key to', openai_key)
if os.path.exists('openai_org.txt'):
    with open('openai_org.txt') as file:
        openai_organization = file.read().strip()
    print('Setting openai.organization to', openai_organization)
    openai.organization = openai_organization


def get_chatgpt_response(post, verbose=False,
                         presence_penalty=0, frequency_penalty=0,
                         num_retries=20, wait=5, version='0301'):
    if verbose:
        print(f'Calling ChatGPT {version}. Input length: {len(post)}')
    while True:
        try:
            ret = openai.ChatCompletion.create(
                model=f"gpt-3.5-turbo-{version}",
                messages=[{"role": "user", "content": post}],
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            break
        except:
            if num_retries == 0:
                raise RuntimeError
            num_retries -= 1
            print(f'Failed to call the ChatGPT API. Wait for {wait} seconds and retry...')
            time.sleep(wait)
            wait *= 2

    return ret


def get_chatgpt_response_content(response):
    assert len(response['choices']) == 1
    return response['choices'][0]['message']['content'].strip()


def openai_ai_text_detect(text, num_retries=20, wait=5):
    thresholds = [10, 45, 90, 98, 100]
    classes = ['very unlikely', 'unlikely', 'unclear if it is',
               'possibly', 'likely']

    while True:
        try:
            ret = openai.Completion.create(
                model="model-detect-v2", prompt=text + '<|disc_score|>',
                max_tokens=1, temperature=1, top_p=1, n=1, logprobs=5,
                stop='\n', stream=False,
            )
            break
        except:
            print('Failed to call the AI Text Classifier API')
            if num_retries == 0:
                raise RuntimeError
            num_retries -= 1
            print(f'Retrying... (wait for {wait} seconds)')
            time.sleep(wait)
            wait *= 2

    score = math.exp(
        ret['choices'][0]['logprobs']['top_logprobs'][0].get('!', -10))
    score = min((1 - score) * 100, 100)
    pred = None
    for i in range(len(thresholds)):
        if score <= thresholds[i]:
            pred = i
            break
    return {
        'score': score,
        'pred': pred,
        'pred_classname': classes[pred],
        'length': len(text),
    }


def get_chatgpt_paraphrase(input, **kwargs):
    post = ('Please paraphrase the following passage, with at least 150 words:'
            f'\n\n{input}')
    response = get_chatgpt_response(post, **kwargs)
    output = get_chatgpt_response_content(response)
    return output


if __name__ == '__main__':
    print('Input for the ChatGPT API (finish by an empty line):')
    text = ''
    while True:
        line = input()
        if len(line) == 0:
            break
        text += line
    print('Generating...')
    response = get_chatgpt_response(text)
    output = get_chatgpt_response_content(response)
    print('Output length:', len(output))
    print('\n')
    print(output)
