import os
from fastapi import FastAPI, Body
import math

import torch
import transformers

from modules import shared
from transformers.generation.logits_process import LogitsProcessorList
from transformers import  set_seed
from modules import scripts, script_callbacks, devices
import gradio as gr



class Model:
    model = None
    tokenizer = None
    logits_bias = None


available_models = []
SEED_LIMIT_NUMPY = 2**32
neg_inf = - 8192.0
current = Model()

base_dir = scripts.basedir()
models_dir = os.path.join(base_dir, "models")


def device():
    return devices.cpu if shared.opts.promptgen_device == 'cpu' else devices.device


def get_model_path():
    dirname = os.path.join(models_dir)
    return dirname



def generate(text,seed, *args):

    path = get_model_path()
    current.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    current.model = transformers.AutoModelForCausalLM.from_pretrained(path)

    assert current.model, 'No model available'
    assert current.tokenizer, 'No tokenizer available'

    current.model.to(device())
    seed = int(seed) % SEED_LIMIT_NUMPY
    set_seed(seed)
    positive_words = open(os.path.join(get_model_path(), 'positive.txt'),
                              encoding='utf-8').read().splitlines()
    positive_words = ['Ġ' + x.lower() for x in positive_words if x != '']
    current.logits_bias = torch.zeros((1, len(current.tokenizer.vocab)), dtype=torch.float32) + neg_inf

    debug_list = []
    for k, v in current.tokenizer.vocab.items():
        if k in positive_words:
            current.logits_bias[0, v] = 0
            debug_list.append(k[1:])
    print(f'Expansion: Vocab with {len(debug_list)} words.')

    text = safe_str(text) + ','
    tokenized_kwargs = current.tokenizer(text, return_tensors="pt")
    tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(device())
    tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(device())
    current_token_length = int(tokenized_kwargs.data['input_ids'].shape[1])
    max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
    max_new_tokens = max_token_length - current_token_length
    features = current.model.generate(**tokenized_kwargs,
                                       top_k=100,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=True,
                                       logits_processor=LogitsProcessorList([logits_processor]))
    response = current.tokenizer.batch_decode(features, skip_special_tokens=True)
    result = safe_str(response[0])
    return {'result':result}


def logits_processor(input_ids, scores):
    assert scores.ndim == 2 and scores.shape[0] == 1
    current.logits_bias = current.logits_bias.to(scores)

    bias = current.logits_bias.clone()
    bias[0, input_ids[0].to(bias.device).long()] = neg_inf
    bias[0, 11] = 0
    return scores + bias

def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace('  ', ' ')
    return x.strip(",. \r\n")



def on_unload():
    current.model = None
    current.tokenizer = None

def prompt_expansion(_: gr.Blocks, app: FastAPI):
    @app.post("/prompt_expansion")
    async def expansion(prompt:str,seed:str):
        result = generate(prompt,seed)
        return  result

script_callbacks.on_app_started(prompt_expansion)
script_callbacks.on_script_unloaded(on_unload)
