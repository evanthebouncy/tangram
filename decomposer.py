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

class INet(nn.Module):

  def __init__(self, anet):
    super(INet, self).__init__()
    # the forward algebraic net
    self.anet = anet

    # self weights
    self.inv_fc1 = nn.Linear(n_hidden, large_hidden)
    self.inv_h_arg1 = nn.Linear(large_hidden, n_hidden)
    self.inv_h_arg2 = nn.Linear(large_hidden, n_hidden)
    self.inv_v_arg1 = nn.Linear(large_hidden, n_hidden)
    self.inv_v_arg2 = nn.Linear(large_hidden, n_hidden)

    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    self.model_loc = './models/tan_inv.mdl'

  def inv_h(self, x):
    ex = self.anet.enc(x)
    ex = F.relu(self.inv_fc1(ex))
    arg1 = F.sigmoid(self.inv_h_arg1(ex))
    arg2 = F.sigmoid(self.inv_h_arg2(ex))
    return arg1, arg2

  def inv_v(self, x):
    ex = self.anet.enc(x)
    ex = F.relu(self.inv_fc1(ex))
    arg1 = F.sigmoid(self.inv_v_arg1(ex))
    arg2 = F.sigmoid(self.inv_v_arg2(ex))
    return arg1, arg2

  def inv_cost(self, op, arg1, arg2, x):
    # ex = self.anet.enc(x)
    pred = op(arg1, arg2)
    pred_dec = self.anet.dec(pred)
    cost = self.anet.auto_enc_cost(x, pred_dec)
    return cost

  # return self.anet.algebra_cost(ex, pred)
    
  # strictly train inv on algebraic space
  def train_inversion(self, ehv_batch):
    self.optimizer.zero_grad()
    # train the algebraic cost
    _, h_batch, v_batch = ehv_batch

    # H operator
    h_arg1, h_arg2, h_result = h_batch
    h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
    h_inv_arg1, h_inv_arg2 = self.inv_h(h_result)
    h_cost = self.inv_cost(self.anet.abstr_h, h_inv_arg1, h_inv_arg2, h_result)

    # V operator
    v_arg1, v_arg2, v_result = v_batch
    v_arg1, v_arg2, v_result = to_torch(v_arg1), to_torch(v_arg2), to_torch(v_result)
    v_inv_arg1, v_inv_arg2 = self.inv_v(v_result)
    v_cost = self.inv_cost(self.anet.abstr_v, v_inv_arg1, v_inv_arg2, v_result)

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
    if i % 100 == 0:
      print ("===== i n v e r s i o n    a e s t h e t i c s ===== ", i)
      print("cost ", inv_cost)
      torch.save(inet.state_dict(), inet.model_loc) 

      bbb, h_batch, v_batch = gen_train_compose_batch()
      bbb = to_torch(bbb)
      # H operator
      h_arg1, h_arg2, h_result = h_batch
      h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
      h_inv_arg1, h_inv_arg2 = inet.inv_h(h_result)
      h_inv_dec1, h_inv_dec2 = anet.dec(h_inv_arg1), anet.dec(h_inv_arg2)
      rec = anet.dec(anet.abstr_h(h_inv_arg1, h_inv_arg2))

      # h_pred = net.abstr_h(net.enc(h_arg1), net.enc(h_arg2))
      # b_rec = net.dec(h_pred)

      for jjj in range(10):
        orig_board = anet.dec_to_board(anet.channel_last(h_result)[jjj])
        arg1_board = anet.dec_to_board(anet.channel_last(h_arg1)[jjj])
        arg2_board = anet.dec_to_board(anet.channel_last(h_arg2)[jjj])
        inv_arg1_board = anet.dec_to_board(h_inv_dec1[jjj])
        inv_arg2_board = anet.dec_to_board(h_inv_dec2[jjj])
        rec_board      = anet.dec_to_board(rec[jjj])

        render_board(orig_board, "inv_board_{}_result.png".format(jjj))
        render_board(arg1_board, "inv_board_{}_arg1.png".format(jjj))
        render_board(arg2_board, "inv_board_{}_arg2.png".format(jjj))
        render_board(inv_arg1_board, "inv_board_{}_inv_arg1.png".format(jjj))
        render_board(inv_arg2_board, "inv_board_{}_inv_arg2.png".format(jjj))
        render_board(rec_board, "inv_board_{}_rec.png".format(jjj))


