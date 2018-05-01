import torch

def accumulate(accum_vals, vals):
    for i in range(len(accum_vals)):
        accum_vals[i].append(vals[i])
    return accum_vals

class BLEU(object):
    def __init__(self, n=4, pad_idx=None):
        self.n = n
        self.pad_idx = pad_idx

        self.reset_corpus()

    def reset_corpus(self):
        self.corpus_match_list, self.corpus_ngram_list = [[] for i in range(4)], [[] for i in range(4)]
        self.corpus_ref_len, self.corpus_hyp_len = [], []

    def pairwise_match_ngram(self, hyp, ref, n):
        # generate ngrams
        if n > 1:
            hyp_ngrams = hyp.unfold(2, n, 1).contiguous() # [bsz x nhyp x lhyp_n x n]
            ref_ngrams = ref.unfold(2, n, 1).contiguous() # [bsz x nref x lref_n x n]
        else:
            hyp_ngrams = hyp.unsqueeze(3).contiguous()
            ref_ngrams = ref.unsqueeze(3).contiguous()

        # compute pairwise similarity
        M_hh = hyp_ngrams[:,:,:,None].eq(hyp_ngrams[:,:,None,:]).sum(-1).int().eq(n).float()[:,:,None] # [bsz x nhyp x   1  x lhyp_n x lhyp_n]
        M_hr = hyp_ngrams[:,:,None,:,None].eq(ref_ngrams[:,None,:,None,:]).sum(-1).int().eq(n).float() # [bsz x nhyp x nref x lhyp_n x lref_n]

        # deal with padding
        if self.pad_idx is not None:
            hyp_mask = hyp_ngrams.eq(self.pad_idx).sum(-1).gt(0) # [bsz x nhyp x lhyp_n]
            ref_mask = ref_ngrams.eq(self.pad_idx).sum(-1).gt(0) # [bsz x nref x lref_n]

            M_hh.masked_fill_(hyp_mask[:,:,None,:,None], 0)
            M_hh.masked_fill_(hyp_mask[:,:,None,None,:], 0)

            M_hr.masked_fill_(hyp_mask[:,:,None,:,None], 0)
            M_hr.masked_fill_(ref_mask[:,None,:,None,:], 0)

        return M_hh, M_hr

    def sent_bleu(self, hyp, ref, smooth=True, inc=False):
        cnt_match_list, cnt_ngram_list = self.match(hyp, ref, smooth=smooth, inc=inc)
        if inc:
            hyp_len, ref_len = self.inc_effective_ref_length(hyp, ref)
        else:
            hyp_len, ref_len = self.effective_ref_length(hyp, ref)
        bleu = self.sent_bleu_ngram(cnt_match_list, cnt_ngram_list, hyp_len, ref_len, inc=inc) # [bsz x nhyp (x lhyp if inc=True)]

        return bleu

    def corpus_bleu(self, reset_corpus=True):
        cnt_match_list = [torch.cat(cnt_match) for cnt_match in self.corpus_match_list]
        cnt_ngram_list = [torch.cat(cnt_ngram) for cnt_ngram in self.corpus_ngram_list]
        hyp_len = torch.cat(self.corpus_hyp_len)
        ref_len = torch.cat(self.corpus_ref_len)
        bleu4, precs, hyplen, reflen = self.corpus_bleu_ngram(cnt_match_list, cnt_ngram_list, hyp_len, ref_len)
        if reset_corpus:
            self.reset_corpus()
        return bleu4, precs, hyplen, reflen

    def add_to_corpus(self, hyp, ref):
        batch_cnt_match, batch_cnt_ngram = self.match(hyp, ref)
        batch_hyp_len, batch_ref_len = self.effective_ref_length(hyp, ref)
        self.corpus_match_list = accumulate(self.corpus_match_list, batch_cnt_match)
        self.corpus_ngram_list = accumulate(self.corpus_ngram_list, batch_cnt_ngram)
        self.corpus_hyp_len.append(batch_hyp_len)
        self.corpus_ref_len.append(batch_ref_len)

    def match(self, hyp, ref, smooth=False, inc=False):
        cnt_match_list, cnt_ngram_list = [], []
        for i in range(1, self.n+1):
            if inc:
                cnt_match, cnt_ngram = self.inc_match_ngram(hyp, ref, i, smooth)
            else:
                cnt_match, cnt_ngram = self.match_ngram(hyp, ref, i, smooth)
            cnt_match_list.append(cnt_match)
            cnt_ngram_list.append(cnt_ngram)
        return cnt_match_list, cnt_ngram_list

    def match_ngram(self, hyp, ref, n, smooth=False):
        """
            hyp : [bsz x nhyp x lhyp] where nhyp is the number of hypothesis (often 1)
            ref : [bsz x nref x lref]
            n   : the n in n-gram

            [Reference]: http://www.phontron.com/class/mtandseq2seq2017/mt-spring2017.chapter10.pdf
        """
        bsz, nhyp, lhyp = hyp.size()
        lhyp_n = lhyp - n + 1
        
        # [Case 1] : hyp or ref is too short to generate n-grams
        if hyp.size(2) < n or ref.size(2) < n:
            cnt_match = hyp.new(1, 1).float().zero_().expand(bsz, nhyp)        # [bsz x nhyp]
        # [Case 2] : hyp and ref are long enough to generate n-grams
        else:
            # [Step 1] : compute pairwise distance ==> `M_hh`, `M_hr`
            M_hh, M_hr = self.pairwise_match_ngram(hyp, ref, n)
            
            # [Step 2] : compute ngram coocurrence between hyp and ref ==> `cnt_match`
            o_hr = M_hr.sum(-1)                                                # [bsz x nhyp x nref x lhyp_n]
            o_hh = M_hh.sum(-1)                                                # [bsz x nhyp x   1  x lhyp_n]
            o_clip, _ = torch.min(o_hh, o_hr).max(dim=2)                       # [bsz x nhyp x lhyp_n]
            cnt_match = torch.sum(o_clip / (1e-16 + o_hh.squeeze(2)), dim=-1)  # [bsz x nhyp]

        ##### Compute `cnt_ngram`
        if self.pad_idx is not None:
            cnt_ngram = (hyp.ne(self.pad_idx).float().sum(-1) - n + 1).clamp(min=0)   # [bsz x nhyp]
        else:
            cnt_ngram = hyp.new(1, 1).float().fill_(max(0, lhyp_n)).expand(bsz, nhyp) # [bsz x nhyp]

        # [Reference]: Chin-Yew Lin and Franz Josef Och (ACL 2004)
        # Automatic evaluation of machine translation quality using longest common
        # subsequence and skip-bigram statistics
        if smooth:
            cnt_match = cnt_match + 1
            cnt_ngram = cnt_ngram + 1

        return cnt_match, cnt_ngram

    def inc_match_ngram(self, hyp, ref, n, smooth=False):
        bsz, nhyp, lhyp = hyp.size()
        lhyp_n = lhyp - n + 1

        # [Case 1] : hyp or ref is too short to generate n-grams
        if hyp.size(2) < n or ref.size(2) < n:
            cnt_match = hyp.new(1, 1, 1).float().zero_().expand(bsz, nhyp, lhyp)         # [bsz x nhyp x lhyp]

        # [Case 2] : hyp and ref are long enough to generate n-grams
        else:
            # [Step 1] : compute pairwise distance ==> `M_hh`, `M_hr`
            M_hh, M_hr = self.pairwise_match_ngram(hyp, ref, n)
            
            # [Step 2] : compute ngram coocurrence between hyp and ref ==> `cnt_match`
            o_hr = M_hr.sum(-1)                                                          # [bsz x nhyp x nref x lhyp_n]
            triu_mask = M_hh.new(lhyp_n, lhyp_n).fill_(1).triu(diagonal=1).byte()        # [lhyp_n x lhyp_n]
            inc_o_hh = M_hh.clone().cumsum(-2).masked_fill_(triu_mask[None,None,:,:], 0) # [bsz x nhyp x   1  x lhyp_n (prefix) x lhyp_n]
            inc_o_hr = o_hr[:,:,:,None,:]                                                # [bsz x nhyp x nref x         1       x lhyp_n]
            inc_o_clip, _ = torch.min(inc_o_hh, inc_o_hr).max(dim=2)                     # [bsz x nhyp x lhyp_n (prefix) x lhyp_n]
            cnt_match = torch.sum(inc_o_clip / (1e-16 + inc_o_hh.squeeze(2)), dim=-1)    # [bsz x nhyp x lhyp_n (prefix)]

            # [Step 3] : pad `cnt_match` into fixed size - [bsz x nhyp x lhyp]
            if n > 1:                                                            
                cnt_pad = cnt_match.new(1, 1, 1).zero_().expand(bsz, nhyp, n-1)          # [bsz x nhyp x n-1]
                cnt_match = torch.cat([cnt_pad, cnt_match], dim=2)                       # [bsz x nhyp x lhyp]

        ##### Compute `cnt_ngram`
        if self.pad_idx is not None:
            cnt_ngram = (hyp.ne(self.pad_idx).float().cumsum(-1) - n + 1).clamp(min=0)   # [bsz x nhyp x lhyp]
        else:
            cnt_ngram = (hyp.new(range(1, lhyp+1)).float() - n + 1).clamp(min=0)
            cnt_ngram = cnt_ngram[None,None,:].expand(bsz, nhyp, lhyp)                   # [bsz x nhyp x lhyp]

        # [Reference]: Chin-Yew Lin and Franz Josef Och (ACL 2004)
        # Automatic evaluation of machine translation quality using longest common
        # subsequence and skip-bigram statistics
        if smooth:
            cnt_match = cnt_match + 1
            cnt_ngram = cnt_ngram + 1

        return cnt_match, cnt_ngram

    def effective_ref_length(self, hyp, ref):
        """
            hyp : [bsz x nhyp x lhyp]
            ref : [bsz x nref x lref]
        """
        bsz, nhyp = hyp.size()[:2]

        if self.pad_idx is not None:
            ref_len = ref.ne(self.pad_idx).float().sum(-1)                                   # [bsz x nref]
            hyp_len = hyp.ne(self.pad_idx).float().sum(-1)                                   # [bsz x nhyp]

            # prefer the shorter reference when there are multiple closest reference lengths
            _, eff_ref_idx = torch.abs(hyp_len[:,:,None] - ref_len[:,None,:] - .01).min(-1)  # [bsz x nhyp] 
            eff_ref_len = ref_len[[[i] for i in range(bsz)], eff_ref_idx]                    # [bsz x nhyp]
        else:
            hyp_len = hyp.new(1, 1).float().fill_(hyp.size(2)).expand(bsz, nhyp)             # [bsz x nhyp]
            eff_ref_len = ref.new(1, 1).float().fill_(ref.size(2)).expand(bsz, nhyp)         # [bsz x nhyp]

        return hyp_len, eff_ref_len

    def inc_effective_ref_length(self, hyp, ref):
        """
            hyp : [bsz x nhyp x lhyp]
            ref : [bsz x nref x lref]
        """
        bsz, nhyp, lhyp = hyp.size()

        if self.pad_idx is not None:
            ref_len = ref.ne(self.pad_idx).float().sum(-1) # [bsz x nref]
            hyp_len = hyp.ne(self.pad_idx).float().cumsum(-1).view(bsz, nhyp * lhyp)          # [bsz x (nhyp * lhyp)]
            
            # prefer the shorter reference when there are multiple closest reference lengths
            _, eff_ref_idx = torch.abs(hyp_len[:,:,None] - ref_len[:,None,:] - .01).min(-1)   # [bsz x (nhyp * lhyp)]
            eff_ref_len = ref_len[[[i] for i in range(bsz)], eff_ref_idx]                     # [bsz x (nhyp * lhyp)]
            
            hyp_len = hyp_len.view(bsz, nhyp, lhyp)                                           # [bsz x nhyp x lhyp]
            eff_ref_len = eff_ref_len.view(bsz, nhyp, lhyp)                                   # [bsz x nhyp x lhyp]
        else:
            hyp_len = hyp.new(range(1, lhyp+1)).float()[None,None,:].expand(bsz, nhyp, lhyp)  # [bsz x nhyp x lhyp]
            eff_ref_len = ref.new(1, 1, 1).float().fill_(ref.size(2)).expand(bsz, nhyp, lhyp) # [bsz x nhyp x lhyp]

        return hyp_len, eff_ref_len

    def corpus_bleu_ngram(self, cnt_match_list, cnt_ngram_list, hyp_len, ref_len):
        bsz, nhyp = cnt_match_list[0].size()
        n = len(cnt_ngram_list)

        hyp_len, ref_len = hyp_len.sum(0), ref_len.sum(0)
        brevity = torch.min(cnt_match_list[0].new(1).zero_(), 1 - ref_len / hyp_len).exp() # [nhyp]

        precs = []
        sum_log_prec = cnt_match_list[0].new(nhyp).zero_() # [nhyp]
        for i in range(n):
            prec = cnt_match_list[i].sum(0) / cnt_ngram_list[i].sum(0)
            sum_log_prec += torch.log(prec)
            precs.append(prec)

        bleu = brevity * torch.exp(sum_log_prec / n) # [nhyp]

        return bleu, precs, hyp_len, ref_len

    def sent_bleu_ngram(self, cnt_match_list, cnt_ngram_list, hyp_len, ref_len, inc=False):
        """
            cnt_match_list : list of [bsz x nhyp x lhyp] | [bsz x nhyp]
            cnt_ngram_list : list of [bsz x nhyp x lhyp] | [bsz x nhyp]
            cnt_hyp_len    : [bsz x nhyp x lhyp]         | [bsz x nhyp]
            ref_len        : [bsz x nhyp]                | [bsz x nhyp]
        """
        
        bsz, nhyp = cnt_match_list[0].size()[:2]
        if inc:
            lhyp = cnt_match_list[0].size(2)
            sum_log_prec = cnt_match_list[0].new(bsz, nhyp, lhyp).zero_() # [bsz x nhyp x lhyp]
        else:
            sum_log_prec = cnt_match_list[0].new(bsz, nhyp).zero_() # [bsz x nhyp]
        n = len(cnt_ngram_list)
        brevity = torch.min(cnt_match_list[0].new(1).zero_(), 1 - ref_len / hyp_len).exp() # [bsz x nhyp (x lhyp if inc=True)]
        for i in range(n):
            prec = cnt_match_list[i] / cnt_ngram_list[i]
            sum_log_prec = sum_log_prec + torch.log(prec)

        bleu = brevity * torch.exp(sum_log_prec / n) # [bsz x nhyp (x lhyp if inc=True)]

        return bleu

