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

class Decompose(nn.Module):

  def __init__(self, encoder):
    super(Deconstruct, self).__init__()

    self.encoder = encoder

    # ====== detect what kind of shape it is, Primitive, or H, or V ====== #
    self.op_detect_fc = nn.Linear(n_hidden, len(OPS))

    # ====== given that it is a primitive . . . predict type and orientation ====== #
    self.dec_primitive = nn.Linear(n_hidden, len(SHAPE_TYPES) * len(ORIENTATIONS))

    # ====== given that it is H . . . predict the two arguments in sequence ====== #
    # the first argument . . .
    self.h_arg1_1 = nn.Linear(n_hidden, n_hidden + n_hidden)
    self.h_arg1_2 = nn.Linear(n_hidden + n_hidden, n_hidden)
    # conditioned on the first argument, the second argument
    # here the first n_hidden + n_hidden is cat (hidden_state, first argument)
    self.h_arg2_1 = nn.Linear(n_hidden + n_hidden, n_hidden + n_hidden)
    self.h_arg2_2 = nn.Linear(n_hidden + n_hidden, n_hidden)

    # ======= given that it is V . . . predict the two arguments in sequence ====== #
    # the first argument . . .
    self.v_arg1_1 = nn.Linear(n_hidden, n_hidden + n_hidden)
    self.v_arg1_2 = nn.Linear(n_hidden + n_hidden, n_hidden)
    # conditioned on the first argument, the second argument
    # here the first n_hidden + n_hidden is cat (hidden_state, first argument)
    self.v_arg2_1 = nn.Linear(n_hidden + n_hidden, n_hidden + n_hidden)
    self.v_arg2_2 = nn.Linear(n_hidden + n_hidden, n_hidden)

    
  def op_detect(self, enc_x):
    the_op = self.op_detect_fc(enc_x)
    print (the_op.size())
    assert 0
    smol_const = to_torch(np.array([1e-6]))
    x = x + smol_const.expand(x.size())
    x = F.softmax(x, dim=2)

if __name__ == '__main__':

  net = ANet().cuda()
  model_loc = './models/tan_algebra.mdl'
  net.load_state_dict(torch.load(model_loc))

  decompose = Decompose(net)

  optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


#  for i in range(100000):
#    optimizer.zero_grad()
#
#    # train for the embedding cost
#    b = gen_train_embed_batch()
#    b = to_torch(b)
#    # this embedding has shape [20 x 20] = [batch x hidden]
#    b_emb = net.enc(b)
#    # this reconstruction should have shape [20 x 6 * 6 x 6]
#    b_rec = net.dec(b_emb)
#    embed_cost = net.auto_enc_cost(b, b_rec)
#
#    # train the algebraic cost
#    h_batch, v_batch = gen_train_compose_batch()
#    # H operator
#    h_arg1, h_arg2, h_result = h_batch
#    h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
#    h_pred = net.abstr_h(net.enc(h_arg1), net.enc(h_arg2))
#    h_result = net.enc(h_result)
#    h_cost = net.algebra_cost(h_result, h_pred)
#
#    # V operator
#    v_arg1, v_arg2, v_result = v_batch
#    v_arg1, v_arg2, v_result = to_torch(v_arg1), to_torch(v_arg2), to_torch(v_result)
#    v_pred = net.abstr_v(net.enc(v_arg1), net.enc(v_arg2))
#    v_result = net.enc(v_result)
#    v_cost = net.algebra_cost(v_result, v_pred)
#
#    alg_cost = h_cost + v_cost
#
#    cost = embed_cost + alg_cost
#    cost.backward()
#    optimizer.step()
#
#    if i % 1000 == 0:
#      print ("===== d i a g n o s t i c    a e s t h e t i c s ===== ", i)
#      print("cost ", cost)
#      print ('embed cost', embed_cost)
#      print ('alg_cost', alg_cost)
#      for jjj in range(20):
#        orig_board = net.dec_to_board(net.channel_last(b)[jjj])
#        rec_board = net.dec_to_board(b_rec[jjj])
#        # print (orig_board)
#        # print (rec_board)
#        render_board(orig_board, "board_{}_orig.png".format(jjj))
#        render_board(rec_board, "board_{}_rec.png".format(jjj))
#
#        torch.save(net.state_dict(), model_loc) 
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
