import os, sys
import time
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../..')
from shared import data, utils, models, metric

parser = argparse.ArgumentParser(description='train_vaml.py')
parser.add_argument('--save_data', default='../data/iwslt14', type=str, help="Input file for the prepared data")
parser.add_argument('--work_dir', default='VAML', type=str, help='Experiment results directory.')
parser.add_argument('--critic_path', default=None, type=str, help='Path to the pretrained critic.')

parser.add_argument('--nemb', type=int, default=256, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=256, help='number of hidden units per layer')
parser.add_argument('--natthid', type=int, default=256, help='number of hidden units for the last rnn layer')
parser.add_argument('--nlayer', type=int, default=1, help='number of layers')
parser.add_argument('--drope', type=float, default=0.0, help='dropout to embedding (0 = no dropout)')
parser.add_argument('--droph', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--att_mode', type=str, default='dotprod', help='type of attention function')
parser.add_argument('--input_feed', action='store_true', help='use input feeding.')

parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--train_bs', type=int, default=42)
parser.add_argument('--valid_bs', type=int, default=50)
parser.add_argument('--test_bs', type=int, default=50)
parser.add_argument('--param_init', type=float, default=0.1, help='Parameters are initialized with U(-param_init, param_init)')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.6, help='initial learning rate')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')

parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay rate')
parser.add_argument('--num_decay', type=int, default=6, help='max number of learning rate decay')

parser.add_argument('--nsample', type=int, default=4, help='number of samples used for training')

parser.add_argument('--vaml_coeff', type=float, default=0.2, help='probability of using samples from the actor')

parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--debug', action='store_true', help='run in debug mode (do not create exp dir). also only use a small fraction of data.')
parser.add_argument('--test_only', action='store_true', help='only run test evaluation.')

parser.add_argument('--beamsize', type=int, default=5, help='size of the beam used for beam search')

parser.add_argument('--ppl_anneal', action='store_true', help="use perplexity as the learing rate annealing metric")

args = parser.parse_args()

if args.critic_path is None:
    args.vaml_coeff = 0.0

# # Set the random seed manually for reproducibility.
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     if not args.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#     else:
#         torch.cuda.manual_seed(args.seed)

if args.test_only:
    ##### init logger
    logging = utils.create_exp_dir(args.work_dir, scripts_to_save=['train_vaml.py'], debug=True)
    logging('==> Testing only mode')

    ##### load vocab
    vocab = torch.load(args.save_data + '-vocab.pt')

    ntoken_src, ntoken_tgt = len(vocab['src']), len(vocab['tgt'])
    src_pad_idx, tgt_pad_idx, tgt_unk_idx, bos_idx, eos_idx = \
        vocab['src'].pad_idx, vocab['tgt'].pad_idx, vocab['tgt'].unk_idx, vocab['tgt'].bos_idx, vocab['tgt'].eos_idx

else:
    ##### init logger
    args.work_dir = os.path.join(args.work_dir, time.strftime("%Y%m%d-%H%M%S"))
    logging = utils.create_exp_dir(args.work_dir, scripts_to_save=['train_vaml.py'], debug=args.debug)

    logging('==> Args')
    for k, v in args.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    ##### load vocab
    vocab = torch.load(args.save_data + '-vocab.pt')

    ntoken_src, ntoken_tgt = len(vocab['src']), len(vocab['tgt'])
    src_pad_idx, tgt_pad_idx, tgt_unk_idx, bos_idx, eos_idx = \
        vocab['src'].pad_idx, vocab['tgt'].pad_idx, vocab['tgt'].unk_idx, vocab['tgt'].bos_idx, vocab['tgt'].eos_idx

    logging('==> Vocabulary')
    logging('    - vocab size: src={}, tgt={}'.format(ntoken_src, ntoken_tgt))
    logging('    - special symbols [src]: {}'.format(', '.join(['{}={}'.format(sym, vocab['src'].get_idx(sym)) for sym in vocab['src'].special])))
    logging('    - special symbols [tgt]: {}'.format(', '.join(['{}={}'.format(sym, vocab['tgt'].get_idx(sym)) for sym in vocab['tgt'].special])))

    ##### train/valid data
    train = torch.load(args.save_data + '-train.pt')
    valid = torch.load(args.save_data + '-valid.pt')

    tr_iter = data.BucketParallelIterator(train['src'], train['tgt'], args.train_bs, src_pad_idx, tgt_pad_idx, 
                                          shuffle=True, cuda=args.cuda)
    va_iter = data.BucketParallelIterator(valid['src'], valid['tgt'], args.valid_bs, src_pad_idx, tgt_pad_idx, 
                                          shuffle=False, cuda=args.cuda, volatile=True)

    ##### intialize models
    if args.critic_path is not None:
        logging('==> Load pretrained critic')
        critic = torch.load(args.critic_path)
        logging('    - critic path: {}'.format(args.critic_path))
        logging('    - critic tau: {}'.format(critic.dec_tau))
        critic.flatten_parameters()
        critic.eval()
        args.tau = critic.decoder.tau
        if args.cuda: critic.cuda()

    actor = models.Seq2Seq(ntoken_src, ntoken_tgt, args.nemb, args.nhid, args.natthid, args.nlayer, args.drope, args.droph, 
                           src_pad_idx, tgt_pad_idx, bos_idx, eos_idx, att_mode=args.att_mode, input_feed=args.input_feed)

    for p in actor.parameters():
        p.data.uniform_(-args.param_init, args.param_init)

    logging('==> Initalize new actorModel')
    logging('    - number of params: {}'.format(sum(p.data.nelement() for p in actor.parameters())))

    if args.cuda: actor.cuda()

    ##### optimizer related
    if args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(actor.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(0.9, 0.999))

bleu_metric = metric.BLEU(4, tgt_pad_idx)

##### training
def train(epoch):
    actor.train()
    start_time = time.time()
    sum_nll, cnt_nll = 0, 0
    sum_cet, cnt_cet = 0, 0
    
    for batch, (src, tgt) in enumerate(tr_iter, start=1):
        tgt_volatile = Variable(tgt.data.clone(), volatile=True)

        # sample corrputed sequence
        seq = Variable(utils.ngram_sample(tgt.data, args.nsample, low=4, high=ntoken_tgt, 
                                          pad_idx=tgt_pad_idx))
        seq = seq.view(seq.size(0), -1)

        # compute importance weight based on sentence bleu
        ref, hyp = utils.prepare_for_bleu(tgt, seq, eos_idx, tgt_pad_idx, tgt_unk_idx)
        sent_bleu = Variable(bleu_metric.sent_bleu(hyp, ref))

        weight = nn.functional.softmax(sent_bleu, 1)
        weight = weight.view(-1)

        # actor log distribution on seq
        log_act_dist = actor(src, seq)

        # non-padding mask
        mask_seq = seq[1:].ne(tgt_pad_idx).float()

        if args.vaml_coeff != 1:
            # negative loss likelihood estimated by the actor
            nll = -log_act_dist.gather(2, seq[1:].unsqueeze(2)).squeeze(2)

            nll_tgt = nll.data.view(nll.size(0), tgt.size(1), -1)[:,:,0]
            mask_tgt = tgt[1:].data.ne(tgt_pad_idx).float()

            sum_nll += (nll_tgt * mask_tgt).sum()
            cnt_nll += mask_tgt.sum()

        if args.vaml_coeff != 0:
            # cross entropy based on the critic 
            crt_dist = critic(tgt_volatile, seq, out_mode=models.PROB)
            cet = -(Variable(crt_dist.data) * log_act_dist).sum(-1)

            sum_cet += cet.data.sum()
            cnt_cet += mask_seq.data.sum()

        if 0 < args.vaml_coeff < 1:
            # # 1 in mask_nll|mask_cet means the corresponding token will NOT be trained by the loss_nll|loss_cet
            # mask_nll = utils.random_mask(seq[1:].data, rate=args.vaml_coeff)
            # mask_cet = (1 - mask_nll)
            # mask_pad = seq[1:].data.eq(tgt_pad_idx)

            # mask_nll = mask_nll | mask_pad
            # mask_cet = mask_cet | mask_pad

            # # apply the masking
            # nll.data.masked_fill_(mask_nll, 0)
            # cet.data.masked_fill_(mask_cet, 0)

            # loss_nll = nll.sum(0)
            # loss_cet = cet.sum(0)

            # 1 in mask_nll|mask_cet means the corresponding token will be trained by the loss_nll|loss_cet
            mask_cet = utils.random_mask(seq[1:].data, rate=args.vaml_coeff)
            mask_nll = (1 - mask_cet)
            mask_nonpad = seq[1:].data.ne(tgt_pad_idx)
            mask_nll = Variable((mask_nll & mask_nonpad).float())
            mask_cet = Variable((mask_cet & mask_nonpad).float())

            loss_nll = (nll * mask_nll).sum(0)
            loss_cet = (cet * mask_cet).sum(0)

            loss = ((loss_nll + loss_cet) * weight).view(tgt.size(1), -1).sum(1).mean(0)

        elif args.vaml_coeff == 0:
            loss = ((nll * mask_seq).sum(0) * weight).view(tgt.size(1), -1).sum(1).mean(0)

        elif args.vaml_coeff == 1:
            loss = ((cet * mask_seq).sum(0) * weight).view(tgt.size(1), -1).sum(1).mean(0)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        gnorm = nn.utils.clip_grad_norm(actor.parameters(), args.grad_clip)
        optimizer.step()

        # logging 
        if batch % args.log_interval == 0:
            cur_loss = sum_nll / cnt_nll
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:4d}/{:4d} batches | lr {:.6f} | ms/batch {:5.1f} | '
                    'loss nll {:5.2f} | loss cet {:5.2f} | ppl {:8.2f} '.format(
                epoch, batch, tr_iter.num_batch(), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, sum_cet / cnt_cet if cnt_cet > 0 else 0., 
                np.exp(cur_loss)))
            start_time = time.time()
            sum_nll, cnt_nll = 0, 0
            sum_cet, cnt_cet = 0, 0

def evaluate(iterator):
    actor.eval()
    sum_nll, cnt_nll = 0, 0
    for batch, (src, tgt) in enumerate(iterator, start=1):

        # get hyp
        hyps, _ = actor.generate(src, k=args.beamsize, n=1)

        # get log_prob
        log_prob = actor(src, tgt)

        # compute masked nll
        tgt_flat = tgt[1:].view(-1, 1)
        masked_nll = -log_prob.view(-1, ntoken_tgt).gather(1, tgt_flat).masked_select(tgt_flat.ne(tgt_pad_idx))

        # accumulate nll
        sum_nll += masked_nll.data.sum()
        cnt_nll += masked_nll.size(0)

        # pytorch_bleu requires size [bsz x nref|nhyp x seqlen]
        ref, hyp = utils.prepare_for_bleu(tgt, hyps, eos_idx, tgt_pad_idx, tgt_unk_idx, exclude_unk=True)

        # add hyp & ref to corpus
        bleu_metric.add_to_corpus(hyp, ref)

    # sanity check
    vis_idx = np.random.randint(0, tgt.size(1))
    logging('===> [SRC]  {}'.format(vocab['src'].convert_to_sent(src[:,vis_idx].contiguous().data.cpu().view(-1), exclude=[src_pad_idx])))
    logging('===> [REF]  {}'.format(vocab['tgt'].convert_to_sent(tgt[1:,vis_idx].contiguous().data.cpu().view(-1), exclude=[tgt_pad_idx, eos_idx])))
    logging('===> [HYP]  {}'.format(vocab['tgt'].convert_to_sent(hyps[1:,vis_idx,0].contiguous().data.cpu().view(-1), exclude=[tgt_pad_idx, eos_idx])))

    ppl = np.exp(sum_nll / cnt_nll)

    bleu4, precs, hyplen, reflen = bleu_metric.corpus_bleu()
    bleu = bleu4[0] * 100
    logging('PPL {:.3f} | BLEU = {:.3f}, {:.1f}/{:.1f}/{:.1f}/{:.1f}, hyp_len={}, ref_len={}'.format(
        ppl, bleu, *[prec[0]*100 for prec in precs], int(hyplen[0]), int(reflen[0])))
    
    return ppl, bleu

if not args.test_only:
    try:
        best_ppl, best_bleu = float('inf'), 0.
        for epoch in range(args.epochs):
            logging('='*89)
            train(epoch)
            logging('='*89)
            curr_ppl, curr_bleu = evaluate(va_iter)

            if (args.ppl_anneal and curr_ppl <= best_ppl) or curr_bleu >= best_bleu:
                # save the new best model
                if not args.debug:
                    logging('Save model with PPL {:.3f} BLEU {:.3f}'.format(curr_ppl, curr_bleu))
                    utils.save_checkpoint(actor, optimizer, args.work_dir, 'best')
                best_ppl, best_bleu = curr_ppl, curr_bleu
            else:
                # anneal learning rate
                curr_lr = optimizer.param_groups[0]['lr']
                next_lr = curr_lr * args.lr_decay
                if next_lr < args.lr * (args.lr_decay ** args.num_decay):
                    logging('=' * 89)
                    logging('Exiting from the learning rate being too small.')
                    logging("Best dev bleu {:.3f}".format(best_bleu))
                    break
                else:
                    logging('Curr: {:.3f} {:.3f}, Best: {:.3f} {:.3f}. Anneal the learning rate {:.6f} --> {:.6f}'.format(
                        curr_ppl, curr_bleu, best_ppl, best_bleu, curr_lr, next_lr))
                    optimizer.param_groups[0]['lr'] = next_lr
    except KeyboardInterrupt:
        logging('=' * 89)
        logging('Exiting from training early')
        logging("Best dev bleu {:.3f}".format(best_bleu))

##### testing
actor = torch.load(os.path.join(args.work_dir, 'model_best.pt'))
actor.flatten_parameters()

test = torch.load(args.save_data + '-test.pt')
te_iter = data.BucketParallelIterator(test['src'], test['tgt'], args.test_bs, src_pad_idx, tgt_pad_idx, 
                                      shuffle=False, cuda=args.cuda, volatile=True)

logging('='*89)
curr_ppl = evaluate(te_iter)
