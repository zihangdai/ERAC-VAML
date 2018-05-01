import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

LOG_PROB = 0
PROB     = 1
LOGIT    = 2

def replicate(tensor, n):
    if isinstance(tensor, tuple):
        return tuple([replicate(t, n) for t in tensor])

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(2).repeat(1, 1, n, 1).view(
            tensor.size(0), tensor.size(1)*n, tensor.size(2))
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(2).repeat(1, 1, n).view(
            tensor.size(0), tensor.size(1)*n)
    elif tensor.dim() == 1:
        tensor = tensor.unsqueeze(1).repeat(1, n)

    return tensor

def topK_2d(score):
    bs, k, ntoken = score.size()
    flat_score = score.view(bs, k*ntoken)
    top_score, top_index = flat_score.topk(k=k, dim=1)
    top_rowid, top_colid = top_index / ntoken, top_index % ntoken

    return top_score, top_rowid, top_colid

def select_hid(hid, batch_id, row_id):
    bs, k = row_id.size()
    new_hid = hid.view(hid.size(0), bs, k, hid.size(2))[:,batch_id.data,row_id.data]
    new_hid = new_hid.view(hid.size(0), bs*k, hid.size(2))

    return new_hid

def check_decreasing(lengths):
    """
        Check whether the lengths tensor are in descreasing order, which is used for variable-length RNN
        - If true, return None
        - Else, return a decreasing lens with two mappings
    """
    lens, order = torch.sort(lengths, 0, True) 
    if torch.ne(lens, lengths).sum() == 0:
        return None
    else:
        _, rev_order = torch.sort(order)

        return lens, Variable(order), Variable(rev_order)

def varlen_rnn_feed(rnn, embed, lengths, hid=None):
    check_res = check_decreasing(lengths)

    if check_res is None:
        lens = lengths
        output = embed
    else:
        lens, order, rev_order = check_res
        output = embed.index_select(1, order)
        if hid is not None:
            if isinstance(hid, tuple):
                hid = tuple([h.index_select(1, order) for h in hid])
            else:
                hid = hid.index_select(1, order)

    packed_inp = rnn_utils.pack_padded_sequence(output, lens.tolist())
    packed_out, new_hid = rnn(packed_inp, hid)
    output, _ = rnn_utils.pad_packed_sequence(packed_out)
    
    if check_res is not None:
        output = output.index_select(1, rev_order)
        if isinstance(new_hid, tuple):
            new_hid = tuple([hid.index_select(1, rev_order) for hid in new_hid])
        else:
            new_hid = new_hid.index_select(1, rev_order)

    return output, new_hid

class Attention(nn.Module):
    def __init__(self, mode, nhid):
        super(Attention, self).__init__()
        self.mode = mode
        self.nhid = nhid

        assert mode in ['mlp', 'dotprod'], 'Unknown attention mode {}'.format(mode)
        if mode == 'mlp':
            self.att_mlp = nn.Sequential(
                nn.Linear(2 * nhid, nhid), nn.Tanh(), 
                nn.Linear(nhid, 1)
            )

    def dotprod_att(self, query, c_key):
        # query -- [qlen x batch x nhid]
        # c_key -- [clen x batch x nhid]
        # att_score -- [clen x qlen x batch]
        att_score = (c_key.unsqueeze(1) * query.unsqueeze(0)).sum(3)
        
        return att_score

    def mlp_att(self, query, c_key):
        # query -- [qlen x batch x nhid]
        # c_key -- [clen x batch x nhid]
        # att_score -- [clen x qlen x batch]

        expand_size = (c_key.size(0), *query.size())
        c_key_expand = c_key.unsqueeze(1).expand(expand_size)
        query_expand = query.unsqueeze(0).expand(expand_size)

        att_score = self.att_mlp(torch.cat([c_key_expand, query_expand], 3)).squeeze(3)

        return att_score

    def forward(self, query, c_key, c_val=None, mask=None):
        if c_val is None: c_val = c_key

        if self.mode == 'dotprod':
            att_score = self.dotprod_att(query, c_key)
        elif self.mode == 'mlp':
            att_score = self.mlp_att(query, c_key)

        if mask is not None:
            att_score.data.masked_fill_(mask.unsqueeze(1), -float('inf'))
        att_prob = F.softmax(att_score, dim=0)

        # att_prob -- [clen x qlen x batch]
        # c_val    -- [clen x batch x nhid]
        # ant_vec  -- [qlen x batch x nhid]
        ant_vec = (att_prob.unsqueeze(3) * c_val.unsqueeze(1)).sum(0)

        return ant_vec, att_prob

class Encoder(nn.Module):
    def __init__(self, ntoken, nemb, nhid, nlayer, drope, droph,
                 pad_idx, bi_enc=True):
        super(Encoder, self).__init__()

        self.ntoken = ntoken
        self.nemb = nemb
        self.nhid = nhid
        self.nlayer = nlayer

        self.drope = drope
        self.droph = droph

        self.bi_enc = bi_enc
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(ntoken, nemb, padding_idx=pad_idx)
        self.drop_emb = nn.Dropout(drope)

        mult = 2 if bi_enc else 1
        self.rnn = nn.LSTM(input_size=nemb, hidden_size=nhid//mult, num_layers=nlayer, dropout=droph, bidirectional=bi_enc)
        self.drop_hid = nn.Dropout(droph)

    def forward(self, src, hid=None):
        src_embed = self.embedding(src)
        src_embed = self.drop_emb(src_embed)

        lengths = src.data.ne(self.pad_idx).int().sum(0)
        enc_out, enc_hid = varlen_rnn_feed(self.rnn, src_embed, lengths, hid)

        pad_mask = src.data.eq(self.pad_idx)

        return enc_out, enc_hid, pad_mask

class Decoder(nn.Module):
    def __init__(self, ntoken, nemb, nhid, natthid, nlayer, drope, droph,
                 pad_idx, bos_idx, eos_idx, nextra=0, att_mode='dotprod', 
                 tau=1.0, input_feed=False, tie_weights=False):
        super(Decoder, self).__init__()

        self.ntoken = ntoken
        self.nemb = nemb
        self.nhid = nhid
        self.natthid = natthid
        self.nextra = nextra
        self.nlayer = nlayer

        self.drope = drope
        self.droph = droph

        self.tau = tau
        self.input_feed = input_feed

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.embedding = nn.Embedding(ntoken, nemb, padding_idx=pad_idx)
        self.drop_emb = nn.Dropout(drope)

        input_size = nemb + nextra + natthid if input_feed else nemb + nextra
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=nhid, num_layers=nlayer, dropout=droph)
        self.drop_hid = nn.Dropout(droph)

        self.src_att = Attention(att_mode, nhid)
        self.concat_proj = nn.Sequential(nn.Linear(2 * nhid, natthid, bias=att_mode=='mlp'), nn.Tanh())

        self.decoder = nn.Linear(natthid, ntoken)
        if tie_weights:
            assert natthid == nemb, 'nemb should equal natthid for tie weights'
            self.decoder.weight = self.tgt_embedding.weight

    def output_layer(self, output, out_mode):
        logit = self.decoder(output)

        if out_mode == LOG_PROB:
            out = F.log_softmax(logit / self.tau, dim=2)
        elif out_mode == PROB:
            out = F.softmax(logit / self.tau, dim=2)
        elif out_mode == LOGIT:
            out = logit

        return out

    def rnn_input_feed(self, rnn_inp, hid, context, pad_mask, att_out=None):
        if att_out is None:
            dec_bsz = hid[0].size(1) if isinstance(hid, tuple) else hid.size(1)
            att_out = Variable(rnn_inp.data.new(1, dec_bsz, self.natthid).fill_(0))

        rnn_outs, att_outs = [], []
        for i in range(rnn_inp.size(0)):
            rnn_out, hid = self.rnn(torch.cat([rnn_inp[i:i+1], att_out], dim=-1), hid)
            att_out, att_prob = self.src_att(rnn_out, context, mask=pad_mask)
            att_out = self.concat_proj(torch.cat([rnn_out, att_out], -1))
            att_out = self.drop_hid(att_out)

            rnn_outs.append(rnn_out)
            att_outs.append(att_out)

        return rnn_outs, att_outs, hid

    def forward(self, tgt, hid, context, pad_mask=None, extra_inp=None, att_out=None, out_mode=LOG_PROB, ret_rnn_out=False, ret_att_out=False):
        tgt_len, bs = tgt.size()[:2]

        tgt_embed = self.embedding(tgt)
        tgt_embed = self.drop_emb(tgt_embed)

        if self.nextra > 0:
            rnn_inp = torch.cat([tgt_embed, extra_inp], dim=-1)
        else:
            rnn_inp = tgt_embed

        if self.input_feed:
            rnn_outs, att_outs, rnn_hid = self.rnn_input_feed(rnn_inp, hid, context, pad_mask, att_out=att_out)
            att_out = torch.cat(att_outs)
            if ret_rnn_out: rnn_out = torch.cat(rnn_outs)
        else:
            rnn_out, rnn_hid = self.rnn(rnn_inp, hid)

            # attention
            att_out, att_prob = self.src_att(rnn_out, context, mask=pad_mask)
            att_out = self.concat_proj(torch.cat([rnn_out, att_out], -1))
            att_out = self.drop_hid(att_out)

        dec_out = self.output_layer(att_out, out_mode)

        ret_list = [dec_out, rnn_hid]

        if ret_rnn_out:
            ret_list.insert(1, rnn_out)

        if ret_att_out:
            ret_list.append(att_out)
        
        return ret_list

    def return_samples(self, tokens, logits, log_probs, rnn_outs, bs, k, out_mode, ret_rnn_out):
        sequence = torch.stack(tokens)                                        # [len x bs x k]
        
        if out_mode == LOGIT:
            seq_dec_out = torch.stack(logits).view(-1, bs*k, self.ntoken)     # [len x bs x ntoken]
        elif out_mode == PROB:
            seq_logits = torch.stack(logits).view(-1, bs*k, self.ntoken)
            seq_dec_out = F.softmax(seq_logits, dim=-1)                       # [len x bs x ntoken]
        elif out_mode == LOG_PROB:
            seq_dec_out = torch.stack(log_probs).view(-1, bs*k, self.ntoken)  # [len x bs x ntoken]
        
        if ret_rnn_out:
            seq_rnn_out = torch.cat(rnn_outs)                                 # [len x bs x nhid]
            return sequence, seq_dec_out, seq_rnn_out
        else:
            return sequence, seq_dec_out

    def sample(self, hid, context, pad_mask, k, max_len, inp=None, eos_mask=None, temperature=1., out_mode=LOG_PROB, ret_rnn_out=False):
        bs = context.size(1)
        replicate_k = functools.partial(replicate, n=k)

        # create <bos> `inp` and range(k) `batch_id`
        if inp is None:
            inp = Variable(context.data.new(1, bs*k).long().fill_(self.bos_idx), volatile=context.volatile)
        else:
            inp = replicate_k(inp)
        batch_id = Variable(replicate_k(inp.data.new(range(bs))), volatile=context.volatile)

        # repeat `hid`, `context`, `pad_mask` for k times
        context, hid, pad_mask = map(replicate_k, [context, hid, pad_mask])
        att_out = None

        # init helping structure `pad_prob`, `eos_mask` for variable-length decoding
        pad_prob = context.data.new(1, self.ntoken).float().fill_(-float('inf'))
        pad_prob[0, self.pad_idx] = 0
        if eos_mask is None:
            eos_mask = context.data.new(bs, k).byte().fill_(0)
        else:
            eos_mask = replicate_k(eos_mask)

        tokens = [inp.clone().view(bs, k)]
        rnn_outs, logits, log_probs = [], [], []
        for i in range(max_len):
            logit, rnn_out, hid, att_out = self.forward(inp, hid, context, pad_mask, att_out=att_out, 
                out_mode=LOGIT, ret_rnn_out=True, ret_att_out=True)
            logit = logit.view(bs, k, self.ntoken)
            log_prob = F.log_softmax(logit / self.tau, dim=-1)

            if eos_mask is not None and eos_mask.sum() > 0:
                log_prob.data.masked_scatter_(eos_mask.unsqueeze(2), pad_prob.expand(eos_mask.sum(), self.ntoken))
            log_prob = log_prob.view(bs*k, self.ntoken)

            # make sure the last token is <eos> if not finished yet
            if i == max_len - 1:
                token = Variable(inp.data.new(bs, k).fill_(self.eos_idx))
                token.data.masked_fill_(eos_mask, self.pad_idx)
                token = token.view(bs * k, 1)
            else:
                prob_rescaled = (log_prob / temperature).exp()
                token = torch.multinomial(prob_rescaled, 1).detach()
            inp = token.view(1, -1)

            eos_mask = eos_mask | token.data.view_as(eos_mask).eq(self.eos_idx)

            # record per-step states
            rnn_outs.append(rnn_out)
            logits.append(logit)
            log_probs.append(log_prob)
            tokens.append(token.view(bs, k))

            if eos_mask.sum() == bs * k:
                break

        return self.return_samples(tokens, logits, log_probs, rnn_outs, bs, k, out_mode, ret_rnn_out)

    def greedy_search(self, hid, context, pad_mask, max_len, inp=None, eos_mask=None, out_mode=LOG_PROB, ret_rnn_out=False):
        bs = context.size(1)

        # create <bos> `inp`
        if inp is None:
            inp = Variable(context.data.new(1, bs).long().fill_(self.bos_idx), volatile=context.volatile)

        att_out = None

        # init helping structure `pad_prob`, `eos_mask` for variable-length decoding
        pad_prob = context.data.new(1, self.ntoken).float().fill_(-float('inf'))
        pad_prob[0, self.pad_idx] = 0
        if eos_mask is None:
            eos_mask = context.data.new(bs).byte().fill_(0)

        tokens = [inp.clone().view(bs, 1)]
        rnn_outs, logits, log_probs = [], [], []
        for i in range(max_len):
            logit, rnn_out, hid, att_out = self.forward(inp, hid, context, pad_mask, att_out=att_out, 
                out_mode=LOGIT, ret_rnn_out=True, ret_att_out=True)
            logit = logit.view(bs, self.ntoken)
            log_prob = F.log_softmax(logit / self.tau, dim=-1)

            if eos_mask is not None and eos_mask.sum() > 0:
                log_prob.data.masked_scatter_(eos_mask.unsqueeze(1), pad_prob.expand(eos_mask.sum(), self.ntoken))

            # make sure the last token is <eos> if not finished yet
            if i == max_len - 1:
                token = Variable(inp.data.new(bs).fill_(self.eos_idx))
                token.data.masked_fill_(eos_mask, self.pad_idx)
            else:
                _, token = log_prob.max(dim=-1)
                token = token.detach()
            inp = token.view(1, -1)

            rnn_outs.append(rnn_out)
            logits.append(logit)
            log_probs.append(log_prob)
            tokens.append(token)

            eos_mask = eos_mask | token.data.eq(self.eos_idx)
            if eos_mask.sum() == bs:
                break

        return self.return_samples(tokens, logits, log_probs, rnn_outs, bs, 1, out_mode, ret_rnn_out)

    def beam_search(self, hid, context, pad_mask, k, n, max_len, inp=None, eos_mask=None, out_mode=LOG_PROB, ret_rnn_out=False):
        bs = context.size(1)
        replicate_k = functools.partial(replicate, n=k)

        # create <bos> `inp` and range(k) `batch_id`
        if inp is None:
            inp = Variable(context.data.new(1, bs*k).long().fill_(self.bos_idx), volatile=context.volatile)
        else:
            inp = replicate_k(inp)
        batch_id = Variable(replicate_k(inp.data.new(range(bs))), volatile=context.volatile)

        # save a one-step `bos` for constructing the final sequence
        bos = inp.clone().view(bs, k)[:,:n]

        # repeat `hid`, `context`, `pad_mask` for k times
        context, hid, pad_mask = map(replicate_k, [context, hid, pad_mask])
        att_out = None

        # init top score
        top_score = Variable(context.data.new(bs, k).float().fill_(-float('inf')), volatile=context.volatile)
        top_score.data[:,0].fill_(0)

        # init helping structure `pad_prob`, `eos_mask` for variable-length decoding
        pad_prob = context.data.new(1, self.ntoken).float().fill_(-float('inf'))
        pad_prob[0, self.pad_idx] = 0
        if eos_mask is None:
            eos_mask = context.data.new(bs, k).byte().fill_(0)
        else:
            eos_mask = replicate_k(eos_mask)

        top_rowids, top_colids = [], []
        rnn_outs_k, logits_k, log_dists_k = [], [], []
        for i in range(max_len):
            logit, rnn_out, hid, att_out = self.forward(inp, hid, context, pad_mask, att_out=att_out, 
                out_mode=LOGIT, ret_rnn_out=True, ret_att_out=True)
            logit = logit.view(bs, k, self.ntoken)
            log_prob = F.log_softmax(logit / self.tau, dim=-1)

            if eos_mask is not None and eos_mask.sum() > 0:
                log_prob.data.masked_scatter_(eos_mask.unsqueeze(2), pad_prob.expand(eos_mask.sum(), self.ntoken))

            score = top_score.unsqueeze(2) + log_prob

            top_score, top_rowid, top_colid = topK_2d(score)

            top_rowids.append(top_rowid)
            top_colids.append(top_colid)

            rnn_outs_k.append(rnn_out)
            logits_k.append(logit)
            log_dists_k.append(log_prob)

            if isinstance(hid, tuple):
                hid = tuple([select_hid(h, batch_id, top_rowid) for h in hid])
            else:
                hid = select_hid(hid, batch_id, top_rowid)
            inp = top_colid.view(1, -1).detach()

            eos_mask = eos_mask.gather(dim=1, index=top_rowid.data) | top_colid.data.eq(self.eos_idx)

            if eos_mask.sum() == bs * k:
                break

        tokens, rnn_outs, logits, log_probs = [], [], [], []
        batch_id = Variable(top_colids[0].data.new(range(bs))[:,None])
        for i in reversed(range(len(top_colids))):
            if i == len(top_colids) - 1:
                sort_score, sort_idx = torch.sort(top_score, dim=1, descending=True)
                sort_score, sort_idx = sort_score[:,:n], sort_idx[:,:n]
            else:
                sort_idx = top_rowids[i+1].gather(dim=1, index=sort_idx)

            token = top_colids[i].gather(dim=1, index=sort_idx)
            tokens.insert(0, token)

            logit = logits_k[i][batch_id, sort_idx]
            logits.insert(0, logit)

            log_dist = log_dists_k[i][batch_id, sort_idx]
            log_probs.insert(0, log_dist)

            rnn_out = rnn_outs_k[i].view(bs, k, self.nhid)[batch_id, sort_idx] # [bs x n x nhid]
            rnn_outs.insert(0, rnn_out)

        tokens.insert(0, bos)

        return self.return_samples(tokens, logits, log_probs, rnn_outs, bs, n, out_mode, ret_rnn_out)

class Seq2Seq(nn.Module):
    def __init__(self, ntoken_src, ntoken_tgt, nemb, nhid, natthid, nlayer, drope, droph,  
                 src_pad_idx, tgt_pad_idx, bos_idx, eos_idx, bi_enc=True, nextra=0, 
                 dec_tau=1.0, att_mode='dotprod', input_feed=False, tie_weights=False):
        super(Seq2Seq, self).__init__()

        self.ntoken_src = ntoken_src
        self.ntoken_tgt = ntoken_tgt
        self.nemb = nemb
        self.nhid = nhid
        self.natthid = natthid
        self.nlayer = nlayer
        self.bi_enc = bi_enc
        self.dec_tau = dec_tau

        self.drope = drope
        self.droph = droph

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        ##### Encoder
        self.encoder = Encoder(ntoken_src, nemb, nhid, nlayer, drope, droph, src_pad_idx)

        ##### Decoder
        self.decoder = Decoder(ntoken_tgt, nemb, nhid, natthid, nlayer, drope, droph, 
                               tgt_pad_idx, bos_idx, eos_idx, nextra=nextra, att_mode=att_mode, 
                               tau=dec_tau, input_feed=input_feed, tie_weights=tie_weights)

        ##### Flatten params
        self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def init_dec_hidden(self, enc_hid):
        if self.bi_enc:
            if isinstance(enc_hid, tuple):
                dec_hid = tuple([torch.cat([hid[0::2], hid[1::2]], 2) for hid in enc_hid])
            else:
                dec_hid = torch.cat([enc_hid[0::2], enc_hid[1::2]], 2)
        else:
            dec_hid = enc_hid

        return dec_hid

    def forward(self, src, tgt, extra_inp=None, out_mode=LOG_PROB, ret_rnn_out=False):
        enc_out, enc_hid, pad_mask = self.encoder(src)
        init_hid = self.init_dec_hidden(enc_hid)

        if tgt.size(1) > src.size(1) and tgt.size(1) % src.size(1) == 0:
            nrep = tgt.size(1) // src.size(1)
            enc_out, init_hid, pad_mask = map(functools.partial(replicate, n=nrep), 
                [enc_out, init_hid, pad_mask])

        if ret_rnn_out:
            dec_out, rnn_out, dec_hid = self.decoder(tgt[:-1], init_hid, enc_out, pad_mask=pad_mask, 
                extra_inp=extra_inp, out_mode=out_mode, ret_rnn_out=ret_rnn_out)

            return dec_out, rnn_out
        else:
            dec_out, dec_hid = self.decoder(tgt[:-1], init_hid, enc_out, pad_mask=pad_mask, 
                extra_inp=extra_inp, out_mode=out_mode)

            return dec_out

    def sample(self, src, k, max_len=100, inp=None, eos_mask=None, temperature=1., out_mode=LOG_PROB):
        enc_out, enc_hid, pad_mask = self.encoder(src)
        dec_hid = self.init_dec_hidden(enc_hid)

        return self.decoder.sample(dec_hid, enc_out, pad_mask, k, max_len=max_len, 
            inp=inp, eos_mask=eos_mask, temperature=temperature, out_mode=out_mode)

    def generate(self, src, k, n=None, max_len=100, inp=None, eos_mask=None, out_mode=LOG_PROB):
        if n is None: n = k
        assert n <= k, 'n should be no larger than k'
        
        enc_out, enc_hid, pad_mask = self.encoder(src)
        dec_hid = self.init_dec_hidden(enc_hid)

        if k == 1:
            return self.decoder.greedy_search(dec_hid, enc_out, pad_mask, max_len, inp, eos_mask, out_mode=out_mode)
        else:
            return self.decoder.beam_search(dec_hid, enc_out, pad_mask, k, n, max_len, inp, eos_mask, out_mode=out_mode)

class SeqRegressor(nn.Module):
    def __init__(self, ntoken, nemb, nhid, natthid, nout, nlayer, drope, droph,
                 pad_idx, nextra=0, att_mode='dotprod', input_feed=False):
        super(SeqRegressor, self).__init__()

        self.ntoken = ntoken
        self.nemb = nemb
        self.nhid = nhid
        self.natthid = natthid
        self.nout = nout
        self.nextra = nextra
        self.nlayer = nlayer

        self.drope = drope
        self.droph = droph

        self.pad_idx = pad_idx
        self.input_feed = input_feed

        self.embedding = nn.Embedding(ntoken, nemb, padding_idx=pad_idx)
        self.drop_emb = nn.Dropout(drope)

        input_size = nemb + nextra + natthid if input_feed else nemb + nextra
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=nhid, num_layers=nlayer, dropout=droph)
        self.drop_hid = nn.Dropout(droph)

        self.src_att = Attention(att_mode, nhid)
        self.concat_proj = nn.Sequential(nn.Linear(2 * nhid, natthid, bias=att_mode=='mlp'), nn.Tanh())

        self.regressor = nn.Linear(natthid, nout)

    def rnn_input_feed(self, rnn_inp, hid, context, pad_mask, att_out=None):
        if att_out is None:
            dec_bsz = hid[0].size(1) if isinstance(hid, tuple) else hid.size(1)
            att_out = Variable(rnn_inp.data.new(1, dec_bsz, self.natthid).fill_(0))

        rnn_outs, att_outs = [], []
        for i in range(rnn_inp.size(0)):
            rnn_out, hid = self.rnn(torch.cat([rnn_inp[i:i+1], att_out], dim=-1), hid)
            att_out, att_prob = self.src_att(rnn_out, context, mask=pad_mask)
            att_out = self.concat_proj(torch.cat([rnn_out, att_out], -1))
            att_out = self.drop_hid(att_out)

            rnn_outs.append(rnn_out)
            att_outs.append(att_out)

        return rnn_outs, att_outs, hid

    def forward(self, tgt, hid, context, pad_mask=None, extra_inp=None, att_out=None, ret_rnn_out=False, ret_att_out=False):
        tgt_len, bs = tgt.size()[:2]

        tgt_embed = self.embedding(tgt)
        tgt_embed = self.drop_emb(tgt_embed)

        if self.nextra > 0:
            rnn_inp = torch.cat([tgt_embed, extra_inp], dim=-1)
        else:
            rnn_inp = tgt_embed

        if self.input_feed:
            rnn_outs, att_outs, rnn_hid = self.rnn_input_feed(rnn_inp, hid, context, pad_mask, att_out=att_out)

            att_out = torch.cat(att_outs)
            if ret_rnn_out: rnn_out = torch.cat(rnn_outs)
        else:
            rnn_out, rnn_hid = self.rnn(rnn_inp, hid)

            att_out, att_prob = self.src_att(rnn_out, context, mask=pad_mask)
            att_out = self.concat_proj(torch.cat([rnn_out, att_out], -1))
            att_out = self.drop_hid(att_out)

        dec_out = self.regressor(att_out)

        ret_list = [dec_out, rnn_hid]

        if ret_rnn_out:
            ret_list.insert(1, rnn_out)

        if ret_att_out:
            ret_list.append(att_out)
        
        return ret_list


class Seq2SeqRegressor(nn.Module):
    def __init__(self, ntoken_src, ntoken_tgt, nemb, nhid, natthid, nout, nlayer, drope, droph, 
                 src_pad_idx, tgt_pad_idx, bi_enc=True, nextra=0, att_mode='dotprod', input_feed=False):
        super(Seq2SeqRegressor, self).__init__()

        self.ntoken_src = ntoken_src
        self.ntoken_tgt = ntoken_tgt
        self.nemb = nemb
        self.nhid = nhid
        self.natthid = natthid
        self.nout = nout
        self.nlayer = nlayer
        self.bi_enc = bi_enc

        self.drope = drope
        self.droph = droph

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        ##### Encoder
        self.encoder = Encoder(ntoken_src, nemb, nhid, nlayer, drope, droph, src_pad_idx)

        ##### Decoder
        self.decoder = SeqRegressor(ntoken_tgt, nemb, nhid, natthid, nout, nlayer, drope, droph, 
                                    tgt_pad_idx, nextra=nextra, att_mode=att_mode, input_feed=input_feed)

        ##### Flatten params
        self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def init_dec_hidden(self, enc_hid):
        if self.bi_enc:
            if isinstance(enc_hid, tuple):
                dec_hid = tuple([torch.cat([hid[0::2], hid[1::2]], 2) for hid in enc_hid])
            else:
                dec_hid = torch.cat([enc_hid[0::2], enc_hid[1::2]], 2)
        else:
            dec_hid = enc_hid

        return dec_hid

    def forward(self, src, tgt, extra_inp=None, ret_rnn_out=False):
        enc_out, enc_hid, pad_mask = self.encoder(src)
        init_hid = self.init_dec_hidden(enc_hid)

        if tgt.size(1) > src.size(1) and tgt.size(1) % src.size(1) == 0:
            nrep = tgt.size(1) // src.size(1)
            enc_out, init_hid, pad_mask = map(functools.partial(replicate, n=nrep), 
                [enc_out, init_hid, pad_mask])

        if ret_rnn_out:
            dec_out, rnn_out, dec_hid = self.decoder(tgt[:-1], init_hid, enc_out, pad_mask=pad_mask, 
                extra_inp=extra_inp, ret_rnn_out=ret_rnn_out)

            return dec_out, rnn_out
        else:
            dec_out, dec_hid = self.decoder(tgt[:-1], init_hid, enc_out, pad_mask=pad_mask, 
                extra_inp=extra_inp)

            return dec_out
