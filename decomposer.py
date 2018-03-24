# takes in a tangram and attempt to decompose it to smaller pieces

from gen import *
from data import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from constants import *

from algebraic_model import *

n_decompose = 8

class INet(nn.Module):

  def __init__(self, anet):
    super(INet, self).__init__()
    # self weights
    self.inv_fc1 = nn.Linear(n_hidden, large_hidden)

    self.inv_h_class = nn.Linear(large_hidden, n_decompose)
    self.inv_h_means = nn.ModuleList([nn.Linear(large_hidden, 2 * n_hidden)\
        for _ in range(n_decompose)])
    self.inv_h_vars  = nn.ModuleList([nn.Linear(large_hidden, 2 * n_hidden)\
        for _ in range(n_decompose)])

    self.inv_v_class = nn.Linear(large_hidden, n_decompose)
    self.inv_v_means = nn.ModuleList([nn.Linear(large_hidden, 2 * n_hidden)\
        for _ in range(n_decompose)])
    self.inv_v_vars  = nn.ModuleList([nn.Linear(large_hidden, 2 * n_hidden)\
        for _ in range(n_decompose)])

    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    self.model_loc = './models/tan_inv.mdl'

    # the forward algebraic net (put it here so self.optimizer don't treat this
    # as part of the parameters which can be problematic
    self.anet = anet

  def inv_h_mixture(self, e):
    '''
    input a goal x (in embedding space)
    output a mixture of gaussian parameters denoting the
    class probability, means, and variance of the decomposition
    '''
    e = F.relu(self.inv_fc1(e))

    # make the class/mixture probabilities and normalize
    class_probs = self.inv_h_class(e)
    # print (class_probs.size())
    class_probs = F.softmax(class_probs, dim=1)
    class_probs = torch.split(class_probs, 1, 1)

    # make the class mean and leave it as is
    class_means = [mu(e) for mu in self.inv_h_means]

    # make the diagonal gaussian variance and make sure it is positive
    # ONE = to_torch(np.array([1.0]))
    # ONE = ONE.expand(class_means[0].size())

    # make sure this is small and positive
    class_vars  = [torch.pow(va(e),2) + 1e-6 for va in self.inv_h_vars]
    # just 1 for now
    class_vars  = [to_torch(np.array([1.0])).expand(class_means[0].size()) for va in self.inv_h_vars]

    return class_probs, class_means, class_vars

  def inv_v_mixture(self, e):
    '''
    input a goal x (in embedding space)
    output a mixture of gaussian parameters denoting the
    class probability, means, and variance of the decomposition
    '''
    e = F.relu(self.inv_fc1(e))

    # make the class/mixture probabilities and normalize
    class_probs = self.inv_v_class(e)
    # print (class_probs.size())
    class_probs = F.softmax(class_probs, dim=1)
    class_probs = torch.split(class_probs, 1, 1)

    # make the class mean and leave it as is
    class_means = [mu(e) for mu in self.inv_v_means]

    # make the diagonal gaussian variance and make sure it is positive
    # ONE = to_torch(np.array([1.0]))
    # ONE = ONE.expand(class_means[0].size())

    # make sure this is small and positive
    class_vars  = [torch.pow(va(e),2) + 1e-6 for va in self.inv_v_vars]
    # just 1 for now
    class_vars  = [to_torch(np.array([1.0])).expand(class_means[0].size()) for va in self.inv_v_vars]

    return class_probs, class_means, class_vars

  def inv_max_likelihood(self, inv_cls, inv_mus, inv_vas):
    # print(inv_cls)
    inv_cls = torch.cat(inv_cls, dim=1)

    print(inv_cls.size())
    _, max_idx = inv_cls.max(1)
    print (max_idx.size())

    inv_mus = [ x1.unsqueeze(0) for x1 in inv_mus]
    inv_mus = torch.cat(inv_mus, dim=0).contiguous().transpose(0,1)
    print (inv_mus.shape)

    max_mus = inv_mus[range(max_idx.size()[0]),max_idx]

    print(max_mus[0])

    print(max_mus.size())

    return torch.split(max_mus, n_hidden, dim=1)
    # arg1 = max_mus.narrow(1,0,n_hidden)
    # arg2 = max_mus.narrow(1,n_hidden,n_hidden)

    # print (arg1.size())
    # print (arg2.size())

    # return arg1, arg2

  def inv_cost(self, inv_cls, inv_mus, inv_vas, arg1_e, arg2_e):
    '''
    compute the log likelihood of data given the generative gaussian mixture model
    as the cost and attempt to reduce the cost here I suppose
    '''
    arg12 = torch.cat((arg1_e, arg2_e), dim=1)
    #print ("cat emb arg size ", arg12.size())
    cl_mu_vas = list(zip(inv_cls, inv_mus, inv_vas))
    #print (len(cl_mu_vas))

    # compute probability for arg12 for each class, and sum it together
    prob_js = []
    for cl_mu_va in cl_mu_vas:
      # probability per classes
      # class_pr * normal(arg12; mu, va)
      cl, mu, va = cl_mu_va
      #print (va)
      cl = cl.contiguous().view(-1)
      #print (cl.size())
      #print (cl)
      norm_const = (1 / torch.sqrt(torch.prod(va, dim=1)))
      #print (norm_const.size())
      #print (norm_const)
      exp_part = torch.sum(-(1/2) * (arg12 - mu) * (1 / va) * (arg12 - mu), dim=1)
      #print (exp_part.size())
      #print (exp_part)
      prob_j = cl * norm_const * torch.exp(exp_part)
      #print (prob_j.size())

      #print (prob_j)
    
      prob_js.append(prob_j)
    
    probs = sum(prob_js)
    #print (probs)
    smol_const = to_torch(np.array([1e-6]))
    probs = probs + smol_const.expand(probs.size())
    #print (probs)
    #print (probs.size())
    neg_log_probs = -torch.sum(torch.log(probs))
    #print (neg_log_probs.size())
    #print (neg_log_probs)
    return neg_log_probs

  # return self.anet.algebra_cost(ex, pred)
    
  # strictly train inv on algebraic space
  def train_inversion(self, ehv_batch):
    enc = self.anet.enc
    self.optimizer.zero_grad()
    # train the algebraic cost
    _, h_batch, v_batch = ehv_batch

    # H operator
    h_arg1, h_arg2, h_result = h_batch
    h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
    h_arg1_e, h_arg2_e, h_result_e = enc(h_arg1), enc(h_arg2), enc(h_result)
    h_inv_cls, h_inv_mus, h_inv_vas = self.inv_h_mixture(h_result_e)
    h_cost = self.inv_cost(h_inv_cls, h_inv_mus, h_inv_vas, h_arg1_e, h_arg2_e)

    # V operator
    v_arg1, v_arg2, v_result = v_batch
    v_arg1, v_arg2, v_result = to_torch(v_arg1), to_torch(v_arg2), to_torch(v_result)
    v_arg1_e, v_arg2_e, v_result_e = enc(v_arg1), enc(v_arg2), enc(v_result)
    v_inv_cls, v_inv_mus, v_inv_vas = self.inv_v_mixture(v_result_e)
    v_cost = self.inv_cost(v_inv_cls, v_inv_mus, v_inv_vas, v_arg1_e, v_arg2_e)


    cost = h_cost + v_cost
    cost.backward()
    self.optimizer.step()

    return cost


if __name__ == '__main__':

  anet = ANet().cuda()
  model_loc = './models/tan_algebra.mdl'
  anet.load_state_dict(torch.load(model_loc))

  inet = INet(anet).cuda()
  for i in range(10001):
    inv_cost = inet.train_inversion(gen_train_compose_batch())
    if i % 400 == 0:
      print ("===== i n v e r s i o n    a e s t h e t i c s ===== ", i)
      print("cost ", inv_cost)
      # gotta make sure this doesn't change cuz we wnat to fix these weigts LUL
      # print("weight from anet ", inet.anet.dec_fc1.weight.data[:1])
      torch.save(inet.state_dict(), inet.model_loc) 

      bbb, h_batch, v_batch = gen_train_compose_batch()
      bbb = to_torch(bbb)
      # H operator
      h_arg1, h_arg2, h_result = h_batch
      h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
      h_cls, h_mus, h_vas = inet.inv_h_mixture(anet.enc(h_result))
      h_inv_arg1, h_inv_arg2 = inet.inv_max_likelihood(h_cls, h_mus, h_vas)

      print (h_inv_arg1.size())
      print (h_inv_arg2.size())
      h_inv_dec1, h_inv_dec2 = anet.dec(h_inv_arg1), anet.dec(h_inv_arg2)
      rec = anet.dec(anet.abstr_h(h_inv_arg1, h_inv_arg2))
      rec2 = anet.dec(anet.abstr_h(anet.enc(h_arg1), anet.enc(h_arg2)))

      # h_pred = net.abstr_h(net.enc(h_arg1), net.enc(h_arg2))
      # b_rec = net.dec(h_pred)

      for jjj in range(10):
        orig_board = anet.dec_to_board(anet.channel_last(h_result)[jjj])
        arg1_board = anet.dec_to_board(anet.channel_last(h_arg1)[jjj])
        arg2_board = anet.dec_to_board(anet.channel_last(h_arg2)[jjj])
        inv_arg1_board = anet.dec_to_board(h_inv_dec1[jjj])
        inv_arg2_board = anet.dec_to_board(h_inv_dec2[jjj])
        rec_board      = anet.dec_to_board(rec[jjj])
        rec2_board      = anet.dec_to_board(rec2[jjj])

        render_board(orig_board, "inv_board_{}_result.png".format(jjj))
        render_board(arg1_board, "inv_board_{}_arg1.png".format(jjj))
        render_board(arg2_board, "inv_board_{}_arg2.png".format(jjj))
        render_board(inv_arg1_board, "inv_board_{}_inv_arg1.png".format(jjj))
        render_board(inv_arg2_board, "inv_board_{}_inv_arg2.png".format(jjj))
        render_board(rec_board, "inv_board_{}_rec.png".format(jjj))
        render_board(rec2_board, "inv_board_{}_rec2.png".format(jjj))

