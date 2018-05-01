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

parser = argparse.ArgumentParser(description='train_erac.py')
parser.add_argument('--save_data', default='../data/iwslt14', type=str, help="Input file for the prepared data")
parser.add_argument('--work_dir', default='ERAC', type=str, help='Experiment results directory.')
parser.add_argument('--actor_path', required=True, type=str, help='Path to the pretrained actor.')
parser.add_argument('--critic_path', required=True, type=str, help='Path to the pretrained critic.')

parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--train_bs', type=int, default=50)
parser.add_argument('--valid_bs', type=int, default=50)
parser.add_argument('--test_bs', type=int, default=50)

parser.add_argument('--optim', default='adam', type=str, help='optimizer to use.')
parser.add_argument('--act_lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--act_beta1', type=float, default=0.0, help='beta1 of Adam')
parser.add_argument('--crt_lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--crt_beta1', type=float, default=0.0, help='beta1 of Adam')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay rate')
parser.add_argument('--num_decay', type=int, default=6, help='max number of learning rate decay')

parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')

parser.add_argument('--smooth_coeff', type=float, default=0.001, help='coefficient for Q value variance regularization')
parser.add_argument('--mle_coeff', type=float, default=0.1, help='coefficient for mle loss')
parser.add_argument('--no_tgtnet', action='store_true', help='not to use target network.')
parser.add_argument('--tgt_speed', type=float, default=0.001, help='coefficient for updating the target network')

parser.add_argument('--nsample', type=int, default=1, help='number of samples used for training')

parser.add_argument('--tau', type=float, default=None, help='temperature in the target distribution')

parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--debug', action='store_true', help='run in debug mode (do not create exp dir). also only use a small fraction of data.')
parser.add_argument('--test_only', action='store_true', help='only run test evaluation.')

parser.add_argument('--beamsize', type=int, default=5, help='size of the beam used for beam search')

parser.add_argument('--ppl_anneal', action='store_true', help="use perplexity as the learing rate annealing metric")

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

if args.test_only:
    ##### init logger
    logging = utils.create_exp_dir(args.work_dir, scripts_to_save=['train_erac.py'], debug=True)
    logging('==> Testing only mode')

    ##### load vocab
    vocab = torch.load(args.save_data + '-vocab.pt')

    ntoken_src, ntoken_tgt = len(vocab['src']), len(vocab['tgt'])
    src_pad_idx, tgt_pad_idx, tgt_unk_idx, bos_idx, eos_idx = \
        vocab['src'].pad_idx, vocab['tgt'].pad_idx, vocab['tgt'].unk_idx, vocab['tgt'].bos_idx, vocab['tgt'].eos_idx

else:
    ##### init logger
    args.work_dir = os.path.join(args.work_dir, time.strftime("%Y%m%d-%H%M%S"))
    logging = utils.create_exp_dir(args.work_dir, scripts_to_save=['train_erac.py'], debug=args.debug)

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
    logging('==> Load pretrained critic')
    critic = torch.load(args.critic_path)
    critic.flatten_parameters()
    logging('    - model path: {}'.format(args.critic_path))
    logging('    - decoder tau: {}'.format(critic.dec_tau))
    if args.tau is None:
        args.tau = critic.dec_tau

    logging('==> Load pretrained actor')
    actor = torch.load(args.actor_path)
    actor.flatten_parameters()
    logging('    - model path: {}'.format(args.actor_path))
    logging('    - decoder tau: {}'.format(actor.dec_tau))

    if args.cuda: 
        critic.cuda()
        actor.cuda()

    if args.use_tgtnet:
        tgt_critic = copy.deepcopy(critic)
        if args.cuda: tgt_critic.cuda()

    ##### optimizer related
    act_optimizer = torch.optim.Adam(actor.parameters(), lr=args.act_lr, betas=(args.act_beta1, 0.999))
    crt_optimizer = torch.optim.Adam(critic.parameters(), lr=args.crt_lr, betas=(args.crt_beta1, 0.999))

##### training metric related
bleu_metric = metric.BLEU(4, tgt_pad_idx)

##### Logging args
logging('==> Args')
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))

##### training
def train_erac(src, tgt):
    ##### Policy execution (actor)
    # sample sequence from the actor
    max_len = min(tgt.size(0) + 10, 50)
    # max_len = min(tgt.size(0) + 5, 50)
    seq, act_log_dist = actor.sample(src, k=args.nsample, max_len=max_len)

    seq = seq.view(seq.size(0), -1)
    mask = seq[1:].ne(tgt_pad_idx).float()
    act_dist = act_log_dist.exp()

    # compute rewards
    ref, hyp = utils.prepare_for_bleu(tgt, seq, eos_idx=eos_idx, pad_idx=tgt_pad_idx, unk_idx=tgt_unk_idx)
    R, bleu = utils.get_rewards(bleu_metric, hyp, ref, return_bleu=True)

    ##### Policy evaluation (critic)
    # compute Q value estimated by the critic
    Q_all = critic(tgt, seq, out_mode=models.LOGIT)

    # compute Q(y_{<t}, y_t)
    Q_mod = Q_all.gather(2, seq[1:].unsqueeze(2)).squeeze(2)

    # compute V_bar(y_{<t})
    act_log_dist.data.masked_fill_(seq.data[1:].eq(tgt_pad_idx)[:,:,None], 0.)

    if args.use_tgtnet:
        tgt_volatile = Variable(tgt.data, volatile=True)
        seq_volatile = Variable(seq.data, volatile=True)
        Q_all_bar = tgt_critic(tgt_volatile, seq_volatile, out_mode=models.LOGIT)

        V_bar = (act_dist.data * (Q_all_bar.data - critic.dec_tau * act_log_dist.data)).sum(2) * mask.data
    else:
        V_bar = (act_dist.data * (Q_all.data - critic.dec_tau * act_log_dist.data)).sum(2) * mask.data

    # compute target value : `Q_hat(s, a) = r(s, a) + V_bar(s')`
    Q_hat = R.clone()
    Q_hat[:-1] += V_bar[1:]

    # compute TD error : `td_error = Q_hat - Q_mod`
    td_error = Variable(Q_hat - Q_mod.data)

    # critic loss
    loss_crt = -td_error * Q_mod
    if args.smooth_coeff > 0:
        loss_crt += args.smooth_coeff * Q_all.var(2)
    loss_crt = loss_crt.sum(0).mean()

    # actor loss
    pg_signal = Q_all.data
    if args.tau > 0:
        # normalize to avoid unstability
        pg_signal -= args.tau * act_log_dist.data / (1e-8 + act_log_dist.data.norm(p=2, dim=2, keepdim=True))
    loss_act = -(Variable(pg_signal) * act_dist).sum(2) * mask
    loss_act = loss_act.sum(0).mean()

    return loss_crt, loss_act, mask, td_error, R, bleu

def train_mle(src, tgt):
    mask = tgt[1:].ne(tgt_pad_idx).float()

    log_act_dist = actor(src, tgt)
    nll = -log_act_dist.gather(2, tgt[1:].unsqueeze(2)).squeeze(2) * mask

    loss = nll.sum(0).mean()

    return loss, nll, mask.data.sum()

def train(epoch):
    actor.train()
    critic.train()
    start_time = time.time()
    sum_nll, sum_res, sum_rwd, sum_bleu = 0, 0, 0, 0
    cnt_nll, cnt_word, cnt_sent = 0, 0, 0
    for batch, (src, tgt) in enumerate(tr_iter, start=1):

        loss_crt, loss_act, mask, td_error, R, bleu = train_erac(src, tgt)

        loss_mle, nll, cnt = train_mle(src, tgt)

        # accumulate nll for computing perplexity (this is not necessary though)
        sum_nll += nll.data.sum()
        sum_rwd += (R * mask.data).sum()
        sum_res += (torch.abs(td_error.data) * mask.data).sum()
        sum_bleu += bleu.sum()
        cnt_nll += cnt 
        cnt_word += mask.data.sum()
        cnt_sent += bleu.nelement()

        # optimization
        act_optimizer.zero_grad()
        (loss_act + args.mle_coeff * loss_mle).backward()
        gnorm_act = nn.utils.clip_grad_norm(actor.parameters(), args.grad_clip)
        act_optimizer.step()

        crt_optimizer.zero_grad()
        loss_crt.backward()
        gnorm_crt = nn.utils.clip_grad_norm(critic.parameters(), args.grad_clip)
        crt_optimizer.step()

        if args.use_tgtnet:
            utils.slow_update(critic, tgt_critic, args.tgt_speed)

        # logging 
        if batch % args.log_interval == 0:
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:4d}/{:4d} batches | lr {:.6f} {:.6f} | ms/batch {:5.1f} | '
                    'ppl {:5.2f} | td error {:.4f} | reward {:.4f} | sent bleu {:6.3f} '.format(
                epoch, batch, tr_iter.num_batch(), act_optimizer.param_groups[0]['lr'],
                crt_optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, 
                np.exp(sum_nll / cnt_nll), sum_res / cnt_word, sum_rwd / cnt_word, 
                sum_bleu / cnt_sent * 100))
            start_time = time.time()
            sum_nll, sum_res, sum_rwd, sum_bleu = 0, 0, 0, 0
            cnt_nll, cnt_word, cnt_sent = 0, 0, 0

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
        ref, hyp = utils.prepare_for_bleu(tgt, hyps, eos_idx=eos_idx, pad_idx=tgt_pad_idx, unk_idx=tgt_unk_idx, exclude_unk=True)
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
        curr_ppl, curr_bleu = evaluate(va_iter)
        for epoch in range(args.epochs):
            logging('='*89)
            train(epoch)
            logging('='*89)
            curr_ppl, curr_bleu = evaluate(va_iter)

            if (args.ppl_anneal and curr_ppl <= best_ppl) or (not args.ppl_anneal and curr_bleu >= best_bleu):
                # save the new best model
                if not args.debug:
                    logging('Save model with PPL {:.3f} BLEU {:.3f}'.format(curr_ppl, curr_bleu))
                    utils.save_checkpoint(actor, act_optimizer, args.work_dir, 'best_actor')
                    utils.save_checkpoint(critic, crt_optimizer, args.work_dir, 'best_critic')
                best_ppl, best_bleu = curr_ppl, curr_bleu
            else:
                # anneal learning rate
                curr_lr = act_optimizer.param_groups[0]['lr']
                next_lr = curr_lr * args.lr_decay
                if next_lr < args.act_lr * (args.lr_decay ** args.num_decay):
                    logging('=' * 89)
                    logging('Exiting from the learning rate being too small.')
                    logging("Best dev bleu {:.3f}".format(best_bleu))
                    break
                else:
                    logging('Curr: {:.3f} {:.3f}, Best: {:.3f} {:.3f}. Anneal the learning rate {:.6f} --> {:.6f}'.format(
                        curr_ppl, curr_bleu, best_ppl, best_bleu, curr_lr, next_lr))
                    act_optimizer.param_groups[0]['lr'] = next_lr
    except KeyboardInterrupt:
        logging('=' * 89)
        logging('Exiting from training early')
        logging("Best dev bleu {:.3f}".format(best_bleu))

##### testing
actor = torch.load(os.path.join(args.work_dir, 'model_best_actor.pt'))
actor.flatten_parameters()

test = torch.load(args.save_data + '-test.pt')
te_iter = data.BucketParallelIterator(test['src'], test['tgt'], args.test_bs, src_pad_idx, tgt_pad_idx, 
                                      shuffle=False, cuda=args.cuda, volatile=True)

logging('='*89)
curr_ppl = evaluate(te_iter)
