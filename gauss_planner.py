from gen import *
from data import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from constants import *
from utils import *

class GNet(nn.Module):
  '''
  the GNet perform planning decomposing a goal into actionable tangrams and assemble!
  captures Non Determinism by trying to do away with it and remember 
  
  ONE TRUE WAE (using RL)

  it holds these following functions:
  enc: takes in the pictoral description and encode it into the latent space
  dec: takes in the latent space and decode it back (use to generate a good embedding)
  abstr_op: takes in two latent variables e1 and e2 compute the forward composition (gives structure in the embedding space)
  decomp: takes in the latent representation and output the decompositions (single gaussian / mean + var)
  '''

  def __init__(self):
    super(GNet, self).__init__()
    # for encding
    # 6 input image channel, 6 output channels, 2x2 square convolution
    self.enc_conv1 = nn.Conv2d(6, 12, 2)
    self.enc_conv2 = nn.Conv2d(12, 12, 2)
    self.enc_fc1 = nn.Linear(192, large_hidden)
    self.enc_fc2 = nn.Linear(large_hidden, n_hidden)

    # for decoding
    self.dec_fc1 = nn.Linear(n_hidden, large_hidden)
    self.dec_fc2 = nn.Linear(large_hidden, L*L*6)

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
    self.inv_h_mu = nn.Linear(large_hidden, 2 * n_hidden)
    self.inv_h_va = nn.Linear(large_hidden, 1)

    self.inv_v_fc = nn.Linear(n_hidden, large_hidden)
    self.inv_v_mu = nn.Linear(large_hidden, 2 * n_hidden)
    self.inv_v_va = nn.Linear(large_hidden, 1)

    self.all_opt = torch.optim.Adam(self.parameters(), lr=0.001)
    self.inv_supervised_opt = torch.optim.Adam(\
      list(self.op_fc.parameters()) +
      list(self.inv_h_fc.parameters()) +
      list(self.inv_h_mu.parameters()) +
      list(self.inv_h_va.parameters()) +
      list(self.inv_v_fc.parameters()) +
      list(self.inv_v_mu.parameters()) +
      list(self.inv_v_va.parameters()),
      lr = 0.001)
    
    self.model_loc = './models/gplanner.mdl'

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
    x = x.view(-1, num_flat_features(x))
    x = F.relu(self.enc_fc1(x))
    e = F.sigmoid(self.enc_fc2(x))
    return e

  def dec(self, x):
    x = self.dec_fc1(x)
    x = self.dec_fc2(x)
    x = x.view(-1, L*L, 6)
    x = F.softmax(x, dim=2)
    smol_const = to_torch(np.array([1e-6]))
    x = x + smol_const.expand(x.size())
    return x

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
    return F.softmax(self.op_fc(e) + 1e-5, dim=1)

  def invert_latent(self, op_type, e):
    '''
    takes in a latent vector e
    takes in the inversion type op_type (H or V)
    outputs a single gaussian distribution of (e1 e2) that decomposes the e:
      mu - the mean for the e1e2 in L^2
      va - the variance 
    '''

    inv_map = {
        'H' : {
            'inv_fc' : self.inv_h_fc,
            'inv_mu' : self.inv_h_mu,
            'inv_va' : self.inv_h_va,
          },
        'V' : {
            'inv_fc' : self.inv_v_fc,
            'inv_mu' : self.inv_v_mu,
            'inv_va' : self.inv_v_va,
          }
        }
    
    # we first enlarge the encoding to a large hidden dimension to get ready
    e_large = F.relu(inv_map[op_type]['inv_fc'](e))
    # produce the mean estimate and variance estimate (variance always positive nonzro)
    mu12 = inv_map[op_type]['inv_mu'](e_large) 
    va12 = torch.pow(inv_map[op_type]['inv_va'](e_large),2) + 0.01 
    return mu12, va12

  def inv_logpr(self, mu12, va12, e1, e2):
    va12 = va12.contiguous().view(-1)
    e12 = torch.cat((e1, e2), dim=1)
    diff = torch.sum(torch.pow(e12 - mu12, 2), dim=1)
    assert diff.size() == va12.size()
    first  = (n_hidden * 2) / 2 * torch.log(va12)
    second = diff / torch.pow(va12, 2)
    logpr  = -(first + second)
    #logpr = -((n_hidden * 2) * torch.log(va12) + diff / torch.pow(va12, 2))
    # if random.random() < 0.01:
    #   print ("diff")
    #   print (diff)
    #   print ("first", first)
    #   print ("second", second)
    #   print ("logpr", logpr)

    # print (logpr.size())
    return logpr
    
  def sample_decompose(self, e):
    '''
    takes in a hidden vector and sample the primitive actions and 
    the latent decompositions (if action is H or V)
    this is only done in test time so batch size is just 1 for now
    '''
    # prediction an action 
    # there are total of 22 actions to predict
    op_pred = self.predict_op(e)
    op_pred = op_pred.data.cpu().numpy()[0]
    op_pred = ACTIONS[np.random.choice(range(len(ACTIONS)), p=op_pred)]

    # if action is atomic / primitive return the action
    if op_pred not in ['H', 'V']:
      return op_pred, None, None

    # otherwise, perform decomposition into the sub-goals
    else:
      # go to numpy sample from the gaussian OH YEAH
      mu, va = self.invert_latent(op_pred, e)
      va = va.expand(mu.size())
      mu, va = mu[0].data.cpu().numpy(), va[0].data.cpu().numpy()
      e1e2 = np.random.normal(mu, va)
      e1e2 = to_torch(e1e2).unsqueeze(0)
      e1, e2 = torch.split(e1e2, n_hidden, dim=1)
      return op_pred, e1, e2
    assert 0, "should not happen blyat!"

  # ====================== TRAINING PROCEDURE =================== #

  # Phase 1: Train a good embedding with algebraic structures
  def train_embedding(self, ehv_batch):
    bb, aa, h_batch, v_batch = ehv_batch

    # enc-dec cost
    bb = to_torch(bb)
    enc_dec_cost = dec_cost(bb, self.dec(self.enc(bb)))

    # H operator
    h_1, h_2, h_target = h_batch
    h_1, h_2, h_target = to_torch(h_1), to_torch(h_2), to_torch(h_target)
    # the forward embedding after applying the H operator
    e_h_pred = self.abstr_op('H', self.enc(h_1), self.enc(h_2))
    h_forward_cost = dec_cost(h_target, self.dec(e_h_pred))

    # V operator
    v_1, v_2, v_target = v_batch
    v_1, v_2, v_target = to_torch(v_1), to_torch(v_2), to_torch(v_target)
    # the forward embedding after applying the V operator
    e_v_pred = self.abstr_op('V', self.enc(v_1), self.enc(v_2))
    v_forward_cost = dec_cost(v_target, self.dec(e_v_pred))

    embedding_cost = enc_dec_cost + h_forward_cost + v_forward_cost

    self.all_opt.zero_grad()
    embedding_cost.backward()
    self.all_opt.step()
    # return the auto-enc cost and the algebraic cost for printing
    return enc_dec_cost, h_forward_cost + v_forward_cost

  # Phase 2: Train a supervised gaussian but fixing the embedding weights 
  def train_supervised(self, ehv_batch):
    bb, aa, h_batch, v_batch = ehv_batch
    # action prediction
    bb, aa = to_torch(bb), to_torch(aa)
    aa_pred = self.predict_op(self.enc(bb))
    action_pred_cost = xentropy_cost(aa, aa_pred)

    # H operator
    h_1, h_2, h = h_batch
    h_1, h_2, h = to_torch(h_1), to_torch(h_2), to_torch(h)
    h_e1, h_e2, h_e = self.enc(h_1), self.enc(h_2), self.enc(h)
    h_mu12, h_va12 = self.invert_latent('H', h_e)
    h_logpr = self.inv_logpr(h_mu12, h_va12, h_e1, h_e2)
    # V operator
    v_1, v_2, v = v_batch
    v_1, v_2, v = to_torch(v_1), to_torch(v_2), to_torch(v)
    v_e1, v_e2, v_e = self.enc(v_1), self.enc(v_2), self.enc(v)
    v_mu12, v_va12 = self.invert_latent('V', v_e)
    v_logpr = self.inv_logpr(v_mu12, v_va12, v_e1, v_e2)

    inversion_cost = -(torch.sum(h_logpr) + torch.sum(v_logpr))
    supervised_cost = action_pred_cost + inversion_cost

    self.inv_supervised_opt.zero_grad()
    supervised_cost.backward()
    self.inv_supervised_opt.step()

    return action_pred_cost, inversion_cost

if __name__ == '__main__':

  print ("WORDS OF ENCOURAGEMENTTT ")
  net = GNet().cuda()
  n_train_emb = 5001
  n_train_sup = 5001
  n_train_RL  = 5001

  # phase 1: train the embedding
  for i in range(n_train_emb):
    c_dec, c_algebra = net.train_embedding(gen_train_planner_batch())
    if i % 1000 == 0:
      print ("===== a l g e b r a i c   e m b   a e s t h e t i c s ===== ", i)
      print("cost dec", c_dec)
      print("cost algebra", c_algebra)
      torch.save(net.state_dict(), net.model_loc) 

  # phase 2: train the supervised inversion (gaussian)
  # -------  to prevent collapse of treasure island, freeze the embeddings
  for i in range(n_train_sup):
    c_pred, c_inv = net.train_supervised(gen_train_planner_batch())
    if i % 1000 == 0:
      print ("===== a l g e b r a i c    a e s t h e t i c s ===== ", i)
      print ("cost action pred ", c_pred)
      print ("cost inv ", c_inv)
      torch.save(net.state_dict(), net.model_loc) 

  # phase 3: trian the RL inversion (specialized gaussian)
  # -------  here we can un-freeze the embeddings
