import os, sys
import time
import argparse
import copy

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../..')
from shared import data, utils, models, metric

parser = argparse.ArgumentParser(description='train_critic.py')
parser.add_argument('--save_data', default='../data/iwslt14', type=str, help="Input file for the prepared data")
parser.add_argument('--work_dir', default='CRITIC', type=str, help='Experiment results directory.')
parser.add_argument('--actor_path', required=True, type=str, help='Path to the pretrained actor.')

parser.add_argument('--nemb', type=int, default=256, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=256, help='number of hidden units per layer')
parser.add_argument('--natthid', type=int, default=256, help='number of hidden units for the last rnn layer')
parser.add_argument('--nlayer', type=int, default=1, help='number of layers')
parser.add_argument('--drope', type=float, default=0.0, help='dropout to embedding (0 = no dropout)')
parser.add_argument('--droph', type=float, default=0.0, help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--att_mode', type=str, default='mlp', help='type of attention function')
parser.add_argument('--input_feed', action='store_true', help='use input feeding.')

parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--debug', action='store_true', help='run in debug mode (do not create exp dir). also only use a small fraction of data.')

parser.add_argument('--train_bs', type=int, default=50)
parser.add_argument('--valid_bs', type=int, default=50)
parser.add_argument('--test_bs', type=int, default=50)

parser.add_argument('--param_init', type=float, default=0.1, help='Parameters are initialized with U(-param_init, param_init)')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--optim', default='adam', type=str, help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--start_decay', type=int, default=0, help='the epoch from which we start lr decay')
parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay rate')
parser.add_argument('--num_decay', type=int, default=5, help='max number of learning rate decay')

parser.add_argument('--no_tgtnet', action='store_true', help='not to use target network.')
parser.add_argument('--tgt_speed', type=float, default=0.001, help='coefficient for updating the target network')
parser.add_argument('--smooth_coeff', type=float, default=0.001, help='coefficient for Q value variance regularization')
parser.add_argument('--nsample', type=int, default=2, help='number of samples used for training')
parser.add_argument('--tau', type=float, default=0.045, help='temperature in the target distribution')

parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')


parser.add_argument('--beamsize', type=int, default=5, help='size of the beam used for beam search')
parser.add_argument('--nretain', type=int, default=1, help='number of sequences to be retained after beam search')

args = parser.parse_args()
args.use_tgtnet = not args.no_tgtnet

# Set the random seed manually for reproducibility.
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     if not args.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#     else:
#         torch.cuda.manual_seed(args.seed)

##### init logger
args.work_dir = os.path.join(args.work_dir, time.strftime("%Y%m%d-%H%M%S"))
logging = utils.create_exp_dir(args.work_dir, scripts_to_save=['train_critic.py'], debug=args.debug)

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
logging('==> Load pretrained actor {}'.format(args.actor_path))
actor = torch.load(args.actor_path)
actor.flatten_parameters()
if args.cuda: actor.cuda()

logging('==> Initilize critic')
critic = models.Seq2Seq(ntoken_tgt, ntoken_tgt, args.nemb, args.nhid, args.natthid, args.nlayer, args.drope, args.droph, 
                        src_pad_idx, tgt_pad_idx, bos_idx, eos_idx, dec_tau=args.tau, att_mode=args.att_mode,
                        input_feed=args.input_feed)

for p in critic.parameters():
    p.data.uniform_(-args.param_init, args.param_init)

if args.cuda: critic.cuda()
logging('    - number of params: {}'.format(sum(p.data.nelement() for p in critic.parameters())))

if args.use_tgtnet:
    logging('==> Initilize target critic')
    tgt_critic = copy.deepcopy(critic)
    if args.cuda: tgt_critic.cuda()

##### optimizer related
if args.optim.lower() == 'sgd':
    optimizer = torch.optim.SGD(critic.parameters(), lr=args.lr)
elif args.optim.lower() == 'adam':
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

##### training metric related
bleu_metric = metric.BLEU(n=4, pad_idx=tgt_pad_idx)

##### training
def train(epoch):
    actor.train()
    critic.train()
    start_time = time.time()
    sum_res, cnt_tok = 0, 0
    sum_score, cnt_sent = 0, 0
    for batch, (src, tgt) in enumerate(tr_iter, start=1):
        # get trajectory
        max_len = min(tgt.size(0) + 5, 50)
        seq, act_log_dist = actor.sample(src, k=args.nsample, max_len=max_len)

        seq = seq.view(seq.size(0), -1).detach()
        mask = seq[1:].ne(tgt_pad_idx).float()

        # compute rewards
        ref, hyp = utils.prepare_for_bleu(tgt, seq, eos_idx=eos_idx, pad_idx=tgt_pad_idx, unk_idx=tgt_unk_idx)
        R, score = utils.get_rewards(bleu_metric, hyp, ref, return_bleu=True)

        # given a demonstration trajectory / real sequence tgt, get Q(y_{<t}, w) for all w in W_+
        Q_all = critic(tgt, seq, out_mode=models.LOGIT)

        # compute Q(y_{<t}, y_t)
        Q_mod = Q_all.gather(2, seq[1:].unsqueeze(2)).squeeze(2)

        # compute V_bar(y_{<t})
        act_log_dist = act_log_dist.data.clone()
        act_log_dist.masked_fill_(seq.data[1:].eq(tgt_pad_idx)[:,:,None], 0.)
        act_dist = act_log_dist.exp()

        if args.use_tgtnet:
            tgt_volatile = Variable(tgt.data, volatile=True)
            seq_volatile = Variable(seq.data, volatile=True)
            Q_all_bar = tgt_critic(tgt_volatile, seq_volatile, out_mode=models.LOGIT)

            if critic.dec_tau > 0:
                V_bar = (act_dist * (Q_all_bar.data - critic.dec_tau * act_log_dist)).sum(2) * mask.data
            else:
                V_bar = (act_dist * Q_all_bar.data).sum(2) * mask.data
        else:
            if critic.dec_tau > 0:
                V_bar = (act_dist * (Q_all.data - critic.dec_tau * act_log_dist)).sum(2) * mask.data
            else:
                V_bar = (act_dist * Q_all.data).sum(2) * mask.data

        # compute target value : `Q_hat(s, a) = r(s, a) + V_bar(s')`
        Q_hat = R.clone()
        Q_hat[:-1] += V_bar[1:]

        # compute TD error : `td_error = Q_hat - Q_mod`
        td_error = Variable(Q_hat - Q_mod.data)

        # construct loss function
        loss = -td_error * Q_mod * mask
        if args.smooth_coeff > 0:
            loss += args.smooth_coeff * Q_all.var(2)
        loss = loss.sum(0).mean() 

        # accumulate nll for computing perplexity (this is not necessary though)
        cnt_tok += mask.data.sum()
        sum_res += (torch.abs(td_error.data) * mask.data).sum()
        cnt_sent += seq.size(1)
        sum_score += score.sum()

        # optimization
        optimizer.zero_grad()
        loss.backward()
        gnorm = nn.utils.clip_grad_norm(critic.parameters(), args.grad_clip)
        optimizer.step()

        if args.use_tgtnet:
            utils.slow_update(critic, tgt_critic, args.tgt_speed)

        # logging 
        if batch % args.log_interval == 0:
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:4d}/{:4d} batches | lr {:.6f} | ms/batch {:5.1f} | '
                    'td error {:5.3f} | score {:5.3f}'.format(
                epoch, batch, tr_iter.num_batch(), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, sum_res / cnt_tok, sum_score / cnt_sent))
            start_time = time.time()
            sum_res, cnt_tok = 0, 0
            sum_score, cnt_sent = 0, 0

def evaluate(iterator):
    # actor.eval()
    critic.eval()
    sum_res, cnt_tok = 0, 0
    for batch, (src, tgt) in enumerate(iterator, start=1):
        max_len = tgt.size(0) + 5
        seq, act_log_dist = actor.sample(src, k=1, max_len=max_len)
        seq = seq.view(seq.size(0), -1).detach()

        # non-padding mask
        mask = seq[1:].ne(tgt_pad_idx).float()

        # given a demonstration trajectory / real sequence tgt, get Q(y_{<t}, w) for all w in W_+
        Q_all = critic(tgt, seq, out_mode=models.LOGIT)

        # compute Q(y_{<t}, y_t)
        Q_mod = Q_all.gather(2, seq[1:].unsqueeze(2)).squeeze(2)                # [tgtlen-1 x bsz]

        # compute V_hat(y_{<t})
        act_log_dist = act_log_dist.data.clone()
        act_log_dist.masked_fill_(seq.data[1:].eq(tgt_pad_idx)[:,:,None], 0.)
        if critic.dec_tau > 0:
            V_hat = (act_log_dist.exp() * (Q_all.data - critic.dec_tau * act_log_dist)).sum(2) * mask.data
        else:
            V_hat = (act_log_dist.exp() * Q_all.data).sum(2) * mask.data

        # compute rewards
        ref, hyp = utils.prepare_for_bleu(tgt, seq, eos_idx=eos_idx, pad_idx=tgt_pad_idx, unk_idx=tgt_unk_idx)
        R = utils.get_rewards(bleu_metric, hyp, ref)
        
        # compute target value : `Q_hat(s, a) = r(s, a) + V_bar(s')`
        Q_hat = R.clone()
        Q_hat[:-1] += V_hat[1:]

        # compute TD error : `td_error = Q_hat - Q_mod`
        td_error = Variable(Q_hat) - Q_mod

        # accumulate nll for computing perplexity (this is not necessary though)
        cnt_tok += mask.data.sum()
        sum_res += (torch.abs(td_error.data) * mask.data).sum()

    res = sum_res / cnt_tok
    logging('Valid td error = {:.5f}'.format(res))
    
    return res

try:
    best_res = float('inf')
    for epoch in range(args.epochs):
        logging('='*89)
        train(epoch)
        logging('='*89)
        curr_res = evaluate(va_iter)
        if curr_res <= best_res:
            # save the new best critic
            if not args.debug:
                logging('Save model with RES {:.5f}'.format(curr_res))
                utils.save_checkpoint(critic, optimizer, args.work_dir, 'best')
            best_res = curr_res
        else:
            if epoch < args.start_decay:
                continue
            # anneal learning rate
            curr_lr = optimizer.param_groups[0]['lr']
            next_lr = curr_lr * args.lr_decay
            if next_lr < args.lr * (args.lr_decay ** args.num_decay):
                logging('=' * 89)
                logging('Exiting from the learning rate being too small.')
                break
            else:
                logging('Curr: {:.3f}, Best: {:.3f}. Anneal the learning rate {:.6f} --> {:.6f}'.format(
                    curr_res, best_res, curr_lr, next_lr))
                optimizer.param_groups[0]['lr'] = next_lr

except KeyboardInterrupt:
    logging('=' * 89)
    logging('Exiting from training early')
logging("Best dev td error {:.3f}".format(best_res))
