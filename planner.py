from gen import *
from data import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from constants import *
from utils import *

n_components = 8

class PNet(nn.Module):
  '''
  the PNet perform planning decomposing a goal into actionable tangrams and assemble!

  it holds these following functions:
  enc: takes in the pictoral description and encode it into the latent space
  invert: takes in the latent representation and output 
  '''

  def __init__(self):
    super(PNet, self).__init__()
    # for encding
    # 6 input image channel, 6 output channels, 2x2 square convolution
    self.enc_conv1 = nn.Conv2d(6, 12, 2)
    self.enc_conv2 = nn.Conv2d(12, 12, 2)
    self.enc_fc1 = nn.Linear(192, large_hidden)
    self.enc_fc2 = nn.Linear(large_hidden, n_hidden)

    # for abstract functions h_abst and v_abst (pulkit regularizer)
    self.h_fc1 = nn.Linear(n_hidden + n_hidden, large_hidden)
    self.h_fc2 = nn.Linear(large_hidden, n_hidden)
    self.v_fc1 = nn.Linear(n_hidden + n_hidden, large_hidden)
    self.v_fc2 = nn.Linear(large_hidden, n_hidden)

    # for decompositions

    # predict the operator, either H or V or one of the primitive shapes
    self.op_fc = nn.Linear(n_hidden, len(ACTIONS))

    # for predicting the e1 e2, one per kind of decomposition
    self.inv_h_fc = nn.Linear(n_hidden, large_hidden)
    self.inv_h_class = nn.Linear(large_hidden, n_components)
    self.inv_h_means = nn.ModuleList([nn.Linear(large_hidden, 2 * n_hidden)\
        for _ in range(n_components)])
    #    self.inv_h_vars  = nn.ModuleList([nn.Linear(large_hidden, 2 * n_hidden)\
    #        for _ in range(n_decompose)])

    self.inv_v_fc = nn.Linear(n_hidden, large_hidden)
    self.inv_v_class = nn.Linear(large_hidden, n_components)
    self.inv_v_means = nn.ModuleList([nn.Linear(large_hidden, 2 * n_hidden)\
        for _ in range(n_components)])
    # self.inv_v_vars  = nn.ModuleList([nn.Linear(large_hidden, 2 * n_hidden)\
    #     for _ in range(n_decompose)])

    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    self.model_loc = './models/planner.mdl'

  # ========================== OPERATORS ============================ #
  def enc(self, x):
    '''
    the x input here is encoded as batch x channel x L x L
    encode into a n_hidden latent representation
    '''
    # Max pooling over a (2, 2) window
    x = F.relu(self.enc_conv1(x))
    # If the size is a square you can only specify a single number
    x = F.relu(self.enc_conv2(x))
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.enc_fc1(x))
    e = F.sigmoid(self.enc_fc2(x))
    return e

  def abstr_op(self, op_type, e1, e2):
    '''
    takes in an op_type either H or V
    takes in two embedded vectors e1 and e2
    algebraically combine them (foward) and produce e
    '''
    op_map = {
        'H' : [self.h_fc1, self.h_fc2],
        'V' : [self.v_fc1, self.v_fc2],
        }
    assert op_type in ['H', 'V']
    assert e1.size() == e2.size()
    e1e2 = torch.cat((e1,e2),1)
    op_fc1 = op_map[op_type][0]
    op_fc2 = op_map[op_type][1]
    e = F.relu(op_fc1(e1e2))
    e = F.sigmoid(op_fc2(e))
    return e

  def predict_op(self, e):
    '''
    takes in a latent vector e
    output a distribution of actions on what is to be done on e:
    one class of primitives / H / V
    '''
    return F.softmax(self.op_fc(e))

  def invert_latent(self, op_type, e):
    '''
    takes in a latent vector e
    takes in the inversion type op_type (H or V)
    output the mixture of gaussian distribution of (e1 e2) that decomposes the e
    outputs: c_pr - the component probabilities (there are n_components)
             mu_i - the mean for the ith component (there are n of them)
             va_i - the variance for ith component (right now it is diagonal of 1s)
    '''

    inv_map = {
        'H' : {
            'inv_fc' : self.inv_h_fc,
            'inv_c' : self.inv_h_class,
            'inv_mu' : self.inv_h_means,
            'inv_va' : 'ignore for now',
          },
        'V' : {
            'inv_fc' : self.inv_v_fc,
            'inv_c' : self.inv_v_class,
            'inv_mu' : self.inv_v_means,
            'inv_va' : 'ignore for now',
          }
        }
    
    # we first enlarge the encoding to a large hidden dimension to get ready
    e = F.relu(inv_map[op_type]['inv_fc'](e))

    # make the class/mixture probabilities and normalize
    class_probs = inv_map[op_type]['inv_c'](e)
    class_probs = F.softmax(class_probs, dim=1)
    class_probs = torch.split(class_probs, 1, 1)

    # make the class mean and leave it as is
    class_means = [mu(e) for mu in inv_map[op_type]['inv_mu']]

    # just 1 for variance for now
    class_vars  = [to_torch(np.array([1.0])).expand(class_means[0].size()) for va in self.inv_h_vars]

    return class_probs, class_means, class_vars
    

  # ====================== TRAINING PROCEDURE =================== #
  def train_algebra(self, ehv_batch):
    self.optimizer.zero_grad()
    # train the algebraic cost
    bb, h_batch, v_batch = ehv_batch

    # Auto Encodings
    b = to_torch(bb)
    # embedding has shape [20 x 20] = [batch x hidden]
    # reconstruction should have shape [20 x 6 * 6 x 6]
    b_rec = self.dec(self.enc(b))
    embed_cost = self.auto_enc_cost(b, b_rec)

    # H operator
    h_arg1, h_arg2, h_result = h_batch
    h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
    h_pred = self.abstr_h(self.enc(h_arg1), self.enc(h_arg2))
    h_pred_dec = self.dec(h_pred)
    h_cost = self.auto_enc_cost(h_result, h_pred_dec)

    # V operator
    v_arg1, v_arg2, v_result = v_batch
    v_arg1, v_arg2, v_result = to_torch(v_arg1), to_torch(v_arg2), to_torch(v_result)
    v_pred = self.abstr_v(self.enc(v_arg1), self.enc(v_arg2))
    v_pred_dec = self.dec(v_pred)
    v_cost = self.auto_enc_cost(v_result, v_pred_dec)

    cost = embed_cost + h_cost + v_cost
    cost.backward()
    self.optimizer.step()

    return embed_cost, h_cost, v_cost

  # ============================ HALPERS ============================ #

  def embed_dist_cost(self, e_target, e):
    diff = e_target - e
    cost_value = torch.sum(torch.pow(diff, 2))
    return cost_value
    

if __name__ == '__main__':

  print ("WORDS OF ENCOURAGEMENTTT ")
  net = PNet().cuda()

  '''
  # train the algebra for some iterations
  for i in range(100001):
    alg_cost = net.train_algebra(gen_train_compose_batch())
    if i % 1000 == 0:
      print ("===== a l g e b r a i c    a e s t h e t i c s ===== ", i)
      print("cost ", alg_cost)
      torch.save(net.state_dict(), net.model_loc) 

      bbb, h_batch, v_batch = gen_train_compose_batch()
      bbb = to_torch(bbb)
      bbb_rec = net.dec(net.enc(bbb))
      # H operator
      h_arg1, h_arg2, h_result = h_batch
      h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
      h_pred = net.abstr_h(net.enc(h_arg1), net.enc(h_arg2))
      b_rec = net.dec(h_pred)

      for jjj in range(10):
        orig_board = net.dec_to_board(net.channel_last(h_result)[jjj])
        arg1_board = net.dec_to_board(net.channel_last(h_arg1)[jjj])
        arg2_board = net.dec_to_board(net.channel_last(h_arg2)[jjj])
        rec_board = net.dec_to_board(b_rec[jjj])
        render_board(arg1_board, "algebra_board_{}_arg1.png".format(jjj))
        render_board(arg2_board, "algebra_board_{}_arg2.png".format(jjj))
        render_board(orig_board, "algebra_board_{}_result.png".format(jjj))
        render_board(rec_board,  "algebra_board_{}_predict.png".format(jjj))

      for jjj in range(10):
        orig_board = net.dec_to_board(net.channel_last(bbb)[jjj])
        rec_board = net.dec_to_board(bbb_rec[jjj])
        render_board(orig_board, "embed_board_{}_orig.png".format(jjj))
        render_board(rec_board, "embed_board_{}_rec.png".format(jjj))
  '''
