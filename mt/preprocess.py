import sys, os
import codecs
import random
import argparse
import gc
import shutil

import numpy as np

import torch

sys.path.append('..')
from shared import *

random.seed(0)

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                     help="Path to the validation target data")
parser.add_argument('-test_src', required=True,
                    help="Path to the tesing source data")
parser.add_argument('-test_tgt', required=True,
                     help="Path to the tesing target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

# Dictionary Options
parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_min_freq', type=int, default=1)
parser.add_argument('-tgt_min_freq', type=int, default=1)

# Truncation options
parser.add_argument('-src_seq_length', type=int, default=50,
                    help="Maximum source sequence length to keep.")
parser.add_argument('-tgt_seq_length', type=int, default=50,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

# Data processing options
parser.add_argument('-shuffle', type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-lower', action='store_true', help='lowercase data')

# 
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

args = parser.parse_args()

def load_data(src_path, tgt_path):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing {} & {} ...'.format(src_path, tgt_path))

    with codecs.open(src_path, 'r', 'utf-8') as src_file, \
         codecs.open(tgt_path, 'r', 'utf-8') as tgt_file:
        while True:
            if args.lower:
                src_symbols = src_file.readline().strip().lower().split()
                tgt_symbols = tgt_file.readline().strip().lower().split()
            else:
                src_symbols = src_file.readline().strip().split()
                tgt_symbols = tgt_file.readline().strip().split()

            if not src_symbols or not tgt_symbols:
                if src_symbols and not tgt_symbols or not src_symbols and tgt_symbols:
                    print('WARNING: source and target do not have the same number of sentences')
                break

            if len(src_symbols) <= args.src_seq_length and len(tgt_symbols) <= args.tgt_seq_length:

                src.append(src_symbols)
                tgt.append(tgt_symbols)
            else:
                ignored += 1

            count += 1

            if count % args.report_every == 0:
                print('... {} sentences prepared'.format(count))

    print('Prepared {} sentences ({} ignored)'.format(count, ignored))

    return src, tgt

def encode_data(data, vocab, tgt):
    tensors = []
    for symbols in data:
        if tgt:
            tensor = vocab.convert_to_tensor(['<bos>'] + symbols + ['<eos>'])
        else:
            tensor = vocab.convert_to_tensor(symbols)
        tensors.append(tensor)

    return tensors

def main():
    vocab = {}
    train_raw, valid_raw, test_raw = {}, {}, {}
    train, valid, test = {}, {}, {}

    print('Load training ...')
    train_raw['src'], train_raw['tgt'] = load_data(args.train_src, args.train_tgt)

    print('Load validation ...')
    valid_raw['src'], valid_raw['tgt'] = load_data(args.valid_src, args.valid_tgt)

    print('Load testing ...')
    test_raw['src'], test_raw['tgt'] = load_data(args.test_src, args.test_tgt)

    print('Build vocabulary ...')
    vocab['src'] = data.Vocab(special=['<pad>', '<unk>'])
    vocab['src'].count_sents(train_raw['src'])
    vocab['src'].build_vocab(min_freq=args.src_min_freq, max_size=args.src_vocab_size)

    vocab['tgt'] = data.Vocab(special=['<pad>', '<unk>', '<bos>', '<eos>'])
    vocab['tgt'].count_sents(train_raw['tgt'])
    vocab['tgt'].build_vocab(min_freq=args.tgt_min_freq, max_size=args.tgt_vocab_size)

    print('Encode data ...')
    for lang in ['src', 'tgt']:
        train[lang] = encode_data(train_raw[lang], vocab[lang], lang=='tgt')
        valid[lang] = encode_data(valid_raw[lang], vocab[lang], lang=='tgt')
        test[lang]  = encode_data(test_raw[lang],  vocab[lang], lang=='tgt')

    print('Output data ...')
    for suffix, raw_data in zip(['train', 'valid', 'test'], [train_raw, valid_raw, test_raw]):
        for lang in ['src', 'tgt']:
            with codecs.open('{}-{}.{}.txt'.format(args.save_data, suffix, lang), 'w', 'utf-8') as f:
                for symbols in raw_data[lang]:
                    f.write('{}\n'.format(' '.join(symbols)))

    for lang in ['src', 'tgt']:
        with codecs.open('{}.vocab.{}.txt'.format(args.save_data, lang), 'w', 'utf-8') as f:
            for sym in vocab[lang].idx2sym:
                f.write('{}\n'.format(sym))

    return vocab, train, valid, test

if __name__ == "__main__":
    vocab, train, valid, test = main()
    gc.collect()
    print("Save data")
    torch.save(vocab, args.save_data + '-vocab.pt')
    torch.save(train, args.save_data + '-train.pt')
    torch.save(valid, args.save_data + '-valid.pt')
    torch.save(test, args.save_data + '-test.pt')
