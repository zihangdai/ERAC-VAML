from collections import Counter, OrderedDict
import os

import numpy as np

from torch.autograd import Variable
import torch

def interleave_keys(a, b):
    """Interleave bits from two sort keys to form a joint sort key.
    Examples that are similar in both of the provided keys will have similar
    values for the key defined by this function. Useful for tasks with two
    text fields like machine translation or natural language inference.
    """
    def interleave(args):
        return ''.join([x for t in zip(*args) for x in t])
    return int(''.join(interleave(format(x, '016b') for x in (a, b))), base=2)

class BucketIterator(object):

    def __init__(self, data, batch_size, pad_id, shuffle=False, batch_first=False, 
                 cuda=False, variable=True, volatile=False):
        """
            data -- list[LongTensor]
        """
        self.data = data
        self.sort_key = lambda x: len(x[0])

        self.num_data = len(self.data)
        self.batch_size = batch_size
        self.pad_id = pad_id

        self.shuffle = shuffle
        self.cache_size = self.batch_size * 20

        self.batch_first = batch_first

        self.cuda = cuda
        self.variable = variable
        self.volatile = volatile

    def _batchify(self, data, pad_id, reverse=False):
        if data[0] is None: return None
        max_len = max(x.size(0) for x in data)

        if self.batch_first:
            batch = data[0].new(len(data), max_len).fill_(pad_id)
            for i in range(len(data)):
                batch[i, :data[i].size(0)].copy_(data[i])
        else:
            batch = data[0].new(max_len, len(data)).fill_(pad_id)
            for i in range(len(data)):
                batch[:data[i].size(0), i].copy_(data[i])

        if self.cuda:
            batch = batch.cuda()
        
        if self.variable:
            batch = Variable(batch, volatile=self.volatile)

        return batch

    def _reset(self):
        self.epoch_indices = np.random.permutation(self.num_data) \
            if self.shuffle else np.array(range(self.num_data))

    def yield_chunk(self, data_list, chunk_size, shuffle_chunk=False):
        offsets = [i for i in range(0, len(data_list), chunk_size)]

        # shuffle chunks insteads of samples in the chunk
        if shuffle_chunk: 
            np.random.shuffle(offsets)

        for offset in offsets:
            yield data_list[offset:offset+chunk_size]

    def __getitem__(self, key):
        return self.data[key]

    def _process_batch(self, batch):
        return self._batchify(batch, self.pad_id)

    def __iter__(self):
        self._reset()
        if self.shuffle:
            for cache_indices in self.yield_chunk(self.epoch_indices, self.cache_size):
                # sort all samples in the cache
                cache = [self.__getitem__(idx) for idx in cache_indices]
                sorted_cache = sorted(cache, key=self.sort_key, reverse=True)
                for batch in self.yield_chunk(sorted_cache, self.batch_size, True):
                    yield self._process_batch(batch)
        else:
            for batch_indices in self.yield_chunk(self.epoch_indices, self.batch_size):
                batch = [self.__getitem__(idx) for idx in batch_indices]
                yield self._process_batch(batch)

    def __len__(self):
        return self.num_data

    def num_batch(self):
        return (self.num_data + self.batch_size - 1) // self.batch_size 

class BucketParallelIterator(BucketIterator):

    def __init__(self, src_data, tgt_data, batch_size, src_pad_id, tgt_pad_id, 
                 shuffle=False, batch_first=False, cuda=False, variable=True, 
                 volatile=False):
        """
            src_data -- list[LongTensor]
            tgt_data -- list[LongTensor] or None
        """
        self.src = src_data
        if tgt_data is not None:
            self.tgt = tgt_data
            assert(len(self.src) == len(self.tgt))
            self.sort_key = lambda x: interleave_keys(len(x[0]), len(x[1]))
        else:
            self.tgt = None
            self.sort_key = lambda x: len(x[0])

        self.num_data = len(self.src)
        self.batch_size = batch_size
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

        self.shuffle = shuffle
        self.cache_size = self.batch_size * 50

        self.batch_first = batch_first

        self.cuda = cuda
        self.variable = variable
        self.volatile = volatile

    def __getitem__(self, key):
        if self.tgt is not None:
            return self.src[key], self.tgt[key]
        else:
            return self.src[key], None

    def _process_batch(self, batch):
        sb, tb = zip(*batch)
        src_batch = self._batchify(sb, self.src_pad_id)
        tgt_batch = self._batchify(tb, self.tgt_pad_id)
        return src_batch, tgt_batch

class Vocab(object):
    def __init__(self, special=['<unk>']):
        self.counter = Counter()
        self.special = special
    
    def load_from_text(self, path):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        for sym in self.special:
            self.add_special(sym)

        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 0 and idx % 200000 == 0:
                    print('    symbol {}'.format(idx))
                sym = line.strip()
                self.add_symbol(sym)

    def count_file(self, path, tokenizer=None):
        print('counting file {} ...'.format(path))
        assert os.path.exists(path)
        if tokenizer is None:
            tokenizer = lambda line: line.strip().split()
        sents = []
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 0 and idx % 200000 == 0:
                    print('    line {}'.format(idx))
                symbols = tokenizer(line)
                sents.append(symbols)
        self.count_sents(sents)

        return sents

    def count_sents(self, sents):
        """
            sents : a list of sentences, where each sent is a list of tokenized symbols
        """
        print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if idx > 0 and idx % 200000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    def build_vocab(self, min_freq=0, max_size=None):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        for sym in self.special:
            self.add_special(sym)
            if sym in self.counter:
                del self.counter[sym]

        for sym, cnt in self.counter.most_common(max_size):
            if cnt < min_freq: break
            self.add_symbol(sym)

    def encode_path(self, path, tokenizer=None, corpus=False):
        print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        if tokenizer is None:
            tokenizer = lambda line: line.strip().split()

        encoded = []
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 0 and idx % 200000 == 0:
                    print('    line {}'.format(idx))
                symbols = tokenizer(line)
                encoded.append(self.convert_to_tensor(symbols))

        if corpus:
            encoded = torch.cat(encoded)

        return encoded

    def encode_sents(self, sents, corpus=False):
        print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if idx > 0 and idx % 200000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if corpus:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)
