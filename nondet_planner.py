from gen import *
from data import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from constants import *
from utils import *

R = 30.0

class NPNet(nn.Module):
  '''
  the NPNet perform planning decomposing a goal into actionable tangrams and assemble!
  captures Non Determinism by trying to do away with it and remember 
  
  ONE TRUE WAE

  it holds these following functions:
  enc: takes in the pictoral description and encode it into the latent space
  invert: takes in the latent representation and output 
  abstr_op: takes in two latent variables e1 and e2 compute the forward composition
  '''

  def __init__(self):
    super(NPNet, self).__init__()
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
    self.inv_h_decomp = nn.Linear(large_hidden, 2 * n_hidden)
    self.inv_h_radius = nn.Linear(large_hidden, 1)

    self.inv_v_fc = nn.Linear(n_hidden, large_hidden)
    self.inv_v_decomp = nn.Linear(large_hidden, 2 * n_hidden)
    self.inv_v_radius = nn.Linear(large_hidden, 1)

    self.island_optimizer = torch.optim.Adam(
      list(self.enc_conv1.parameters())+
      list(self.enc_conv2.parameters())+
      list(self.enc_fc1.parameters())+
      list(self.enc_fc2.parameters())+
      list(self.h_fc1.parameters())+
      list(self.h_fc2.parameters())+
      list(self.v_fc1.parameters())+
      list(self.v_fc2.parameters())+
      list(self.op_fc.parameters())+
      list(self.inv_h_fc.parameters())+
      list(self.inv_h_decomp.parameters())+
      list(self.inv_v_fc.parameters())+
      list(self.inv_v_decomp.parameters()),
      lr=0.001)

    # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    self.inv_optimizer = torch.optim.Adam(
      list(self.inv_h_fc.parameters())+
      list(self.inv_h_decomp.parameters())+
      list(self.inv_h_radius.parameters())+
      list(self.inv_v_fc.parameters())+
      list(self.inv_v_decomp.parameters())+
      list(self.inv_v_radius.parameters()),
      lr=0.001)
    self.model_loc = './models/nplanner.mdl'

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
    return F.softmax(self.op_fc(e), dim=1)

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
            'inv_decomp' : self.inv_h_decomp,
            'inv_radius' : self.inv_h_radius,
          },
        'V' : {
            'inv_fc' : self.inv_v_fc,
            'inv_decomp' : self.inv_v_decomp,
            'inv_radius' : self.inv_v_radius,
          }
        }
    
    # we first enlarge the encoding to a large hidden dimension to get ready
    e_large = F.relu(inv_map[op_type]['inv_fc'](e))
    # decode out the decomposition and radius estimate
    e1e2 = inv_map[op_type]['inv_decomp'](e_large) 
    r = inv_map[op_type]['inv_radius'](e_large) + R

    return e1e2, r
    
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
      e1e2, r = self.invert_latent(op_pred, e)
      print ("radius 4Head ", r)
      e1, e2 = torch.split(e1e2, n_hidden, dim=1)
      return op_pred, e1, e2
    assert 0, "should not happen blyat!"

  # ====================== TRAINING PROCEDURE =================== #
  '''
  the given cost is considered
  for every node in a tangram treeC . . .
  op_pred(enc(x)) = op  (either primitive or composition operator)
  for every composition x = x1 op x2 . . .
  try to latch on to a particular decomposition
  enforce forward function abstr_op(op, enc(x1), enc(x2)) = enc(x)
  '''

  def inv_cost(self, e1e2, r, arg1_e, arg2_e):
    '''
    take in a proposed decomposition e1e2 and the truth arg1_e arg2_e
    return the e1e2 update cost and the radius cost
    conditioned on how close e1e2 are to the arg1_e,arg2_e
    '''
    arg12 = torch.cat((arg1_e, arg2_e), dim=1)
    diff_norm = (arg12 - e1e2).norm(dim=1)
    #print ("diff norm ", diff_norm)
    #print ("radius ", r)

    diff_norm_np = diff_norm.data.cpu().numpy()
    r_np = r.view(-1).data.cpu().numpy()
    within_r = diff_norm_np < r_np
    #print ("within r ", within_r)

    # set up the radius target
    r_np = r.view(-1).data.cpu().numpy()
    bigger_r = r_np + 0.001
    smaller_r = r_np * 0.999
    #print ("bigger r ", bigger_r)
    #print ("smaller r ", smaller_r)
    #print ("within r ", within_r)
    r_target = np.where(within_r, smaller_r, bigger_r)
    r_target = to_torch(r_target).unsqueeze(1)
    #print (r_target)

    # set up the treasure target
    arg12_np = arg12.data.cpu().numpy()
    e1e2_np = e1e2.data.cpu().numpy()
    within_r_extend = np.expand_dims(within_r, axis=1)
    within_r_extend = np.repeat(within_r_extend, n_hidden * 2, axis=1)
    # if random.random() < 0.01:
    #   print (within_r_extend)
    # if the treasure is within the radius r, we go to the arg12 target, else dont move
    arg12_target = to_torch(np.where(within_r_extend, arg12_np, e1e2_np))
    #print (e1e2)
    #print (arg12)
    #print (arg12_target)

    # treasure_cost = torch.sum((arg12_target- e1e2).norm(dim=1) / e1e2.norm(dim=1))
    treasure_cost = torch.sum((arg12_target- e1e2).norm(dim=1))
    radius_cost = self.embed_dist_cost(r_target, r)
    # print ("treasure cost ", treasure_cost)
    # print ("radius cost ", radius_cost)
    # print (torch.sum(diff_norm), radius_cost)

    # return torch.sum(diff_norm) + radius_cost
    return torch.sum(diff_norm), treasure_cost + radius_cost
    return treasure_cost + radius_cost

  def embed_dist_cost(self, e_target, e):
    diff = e_target - e
    # cost_value = torch.sum(torch.pow(diff, 2))
    cost_value = torch.sum(diff.norm(dim=1))
    return cost_value

  def op_cost(self, op_target, op_pred):
    assert op_target.size() == op_pred.size()
    logged_op_pred = torch.log(op_pred)
    cost_value = -torch.sum(op_target * logged_op_pred)
    return cost_value

  def get_costs(self, ehv_batch):
    # batch of tangrams, corresponding actions, h_batch and v_batch
    bb, aa, h_batch, v_batch = ehv_batch

    # Encoding and action production
    bb, aa = to_torch(bb), to_torch(aa)
    aa_pred = self.predict_op(self.enc(bb))
    pred_cost = self.op_cost(aa, aa_pred)

    # H operator
    h_arg1, h_arg2, h_result = h_batch
    h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
    e_h_arg1, e_h_arg2, e_h_result = self.enc(h_arg1), self.enc(h_arg2), self.enc(h_result)
    # step 1: the forward cost of the H operator
    h_pred = self.abstr_op('H', e_h_arg1, e_h_arg2)
    h_forward_cost = self.embed_dist_cost(self.enc(h_result), h_pred)
    # step 2: the reverse cost of the H operator
    h_inv_e1e2, h_inv_r = self.invert_latent('H', e_h_result)
    h_inv_cost, h_inv_cost_pl = self.inv_cost(h_inv_e1e2, h_inv_r, e_h_arg1, e_h_arg2)

    # V operator
    v_arg1, v_arg2, v_result = v_batch
    v_arg1, v_arg2, v_result = to_torch(v_arg1), to_torch(v_arg2), to_torch(v_result)
    e_v_arg1, e_v_arg2, e_v_result = self.enc(v_arg1), self.enc(v_arg2), self.enc(v_result)
    # step 1: the forward cost of the V operator
    v_pred = self.abstr_op('V', e_v_arg1, e_v_arg2)
    v_forward_cost = self.embed_dist_cost(self.enc(v_result), v_pred)
    # step 2: the reverse cost of the V operator
    v_inv_e1e2, v_inv_r = self.invert_latent('V', e_v_result)
    v_inv_cost, v_inv_cost_pl = self.inv_cost(v_inv_e1e2, v_inv_r, e_v_arg1, e_v_arg2)

    # return all the costs
    return pred_cost, h_forward_cost + v_forward_cost, h_inv_cost + v_inv_cost,\
           h_inv_cost_pl + v_inv_cost_pl
  #return pred_cost, h_forward_cost, h_inv_cost, v_forward_cost, v_inv_cost

  def train_embedding(self, ehv_batch):
    self.island_optimizer.zero_grad()
    pred_cost , forward_cost , inv_cost, inv_cost_pl = self.get_costs(ehv_batch)
    cost = pred_cost + forward_cost + inv_cost
    cost.backward()
    self.island_optimizer.step()
    return pred_cost, forward_cost, inv_cost, inv_cost_pl

  def train_planning(self, ehv_batch):
    self.inv_optimizer.zero_grad()
    pred_cost , forward_cost , inv_cost, inv_cost_pl = self.get_costs(ehv_batch)
    cost = inv_cost_pl
    cost.backward()
    self.inv_optimizer.step()
    return pred_cost, forward_cost, inv_cost, inv_cost_pl

  # ============================ HALPERS ============================ #
    

if __name__ == '__main__':

  print ("WORDS OF ENCOURAGEMENTTT ")
  net = NPNet().cuda()
  xx, actions, h_batch, v_batch = gen_train_planner_batch()

  xx, actions = to_torch(xx), to_torch(actions)

  e = net.enc(xx)

  res = net.sample_decompose(e)
  print (res)

  # train the embedding
  for i in range(10001):
    c_prd, c_for, c_inv, c_inv_pl = net.train_embedding(gen_train_planner_batch())
    if i % 1000 == 0:
      print ("===== a l g e b r a i c    a e s t h e t i c s ===== ", i)
      print("cost prd", c_prd)
      print("cost forw", c_for)
      print("cost inv", c_inv)
      print("cost inv pl", c_inv_pl)
      torch.save(net.state_dict(), net.model_loc) 
  # train the inversion
  for i in range(10001):
    c_prd, c_for, c_inv, c_inv_pl = net.train_planning(gen_train_planner_batch())
    if i % 1000 == 0:
      print ("===== a l g e b r a i c    a e s t h e t i c s ===== ", i)
      print("cost prd", c_prd)
      print("cost forw", c_for)
      print("cost inv", c_inv)
      print("cost inv pl", c_inv_pl)
      torch.save(net.state_dict(), net.model_loc) 
