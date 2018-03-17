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

def test_1step(anet, b, xxx):
  b = to_torch(b[1:2, :])
  print ("did we do it boys?")
  b_board    = anet.dec_to_board(anet.channel_last(b)[0])
  render_board(b_board,    "{}_decompose_target.png".format(xxx))

  arg1, arg2, rec = anet.h_decompose(b)

  print (arg1.size())
  print (arg2.size())
  print (rec.size())

  arg1_board = anet.dec_to_board(arg1[0])
  arg2_board = anet.dec_to_board(arg2[0])
  rec_board = anet.dec_to_board(rec[0])
  render_board(arg1_board, "{}_decompose_arg1.png".format(xxx))
  render_board(arg2_board, "{}_decompose_arg2.png".format(xxx))
  render_board(rec_board,  "{}_decompose_rec.png".format(xxx))

if __name__ == '__main__':

  anet = ANet().cuda()
  model_loc = './models/tan_algebra.mdl'
  anet.load_state_dict(torch.load(model_loc))

  b, _, _ = gen_train_compose_batch(1)
  for i in range(10):
    test_1step(anet, b, i)

