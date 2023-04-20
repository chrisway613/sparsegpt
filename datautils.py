import random

import torch
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer_cache_dir=None, data_cache_dir=None, test=False):
    if test:
        return get_wikitext2_testloader(
            nsamples, seed, seqlen, model,
            tokenizer_cache_dir=tokenizer_cache_dir, data_cache_dir=data_cache_dir
        )
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=tokenizer_cache_dir, use_fast=False)

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen

        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100

        trainloader.append((inp, tar))

    return trainloader, testenc


def get_wikitext2_testloader(nsamples, seed, seqlen, model, tokenizer_cache_dir=None, data_cache_dir=None):
    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=tokenizer_cache_dir)

    testdata = load_dataset(
        'wikitext', 'wikitext-2-raw-v1', split='test',
        cache_dir=data_cache_dir, use_auth_token=True
    )
    testenc = tokenizer(" ".join(testdata['text']), return_tensors='pt')

    i, testloader = 0, []
    while True:
        j = i + seqlen
        if j > testenc.input_ids.size(1):
            break

        inp = testenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        testloader.append((inp, tar))

        i = j

    print(f"Total {len(testloader)} samples.\n")
    return testloader


def get_ptb(nsamples, seed, seqlen, model):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen

        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        
        trainloader.append((inp, tar))

    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)

    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen

        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_wikipedia(nsamples, seed, seqlen, model, debug=False):
    print("Loading dataset..")
    train_data = load_dataset(
        'bigscience-data/roots_en_wikipedia', 'en',
        cache_dir='/ssd1/datasets/wikipedia',
        split=f"train[5%:]",
        use_auth_token=True
    )
    if debug:
        train_data = train_data.select(range(1000))
        print("1000 train samples selected")

    val_data = load_dataset(
        'bigscience-data/roots_en_wikipedia', 'en',
        cache_dir='/ssd1/datasets/wikipedia',
        split=f"train[:5%]",
        use_auth_token=True
    ).select(range(3000))
    print("Done!\n")

    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir='/ssd1/models/bloom')

    print("Tokenize..\n")
    trainenc = tokenizer(" ".join(train_data['text']), return_tensors='pt')
    testenc = tokenizer(" ".join(val_data['text']), return_tensors='pt')
    print("Done!\n")
    # TODO: for quick verification, select 1000 samples tmp
    testenc.input_ids = testenc.input_ids[:, :(1000 * seqlen)]

    random.seed(seed)

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen

        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100

        trainloader.append((inp, tar))

    return trainloader, testenc


def get_loaders(name, nsamples=128, seed=42, seqlen=2048, model='', tokenizer_cache_dir=None, data_cache_dir=None, debug=False, test=False):
    if 'wikitext2' in name:
        return get_wikitext2(
            nsamples, seed, seqlen, model, 
            tokenizer_cache_dir=tokenizer_cache_dir, 
            data_cache_dir=data_cache_dir, test=test
        )
    
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model)
    
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model)
    
    if 'wikipedia' in name:
        return get_wikipedia(nsamples, seed, seqlen, model, debug=debug)
