from gen import *
from data import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from constants import *

large_hidden = 6 * n_hidden

def to_torch(x, req = False):
  x = Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), requires_grad = req)
  return x

class ANet(nn.Module):

  def __init__(self):
    super(ANet, self).__init__()
    # for encding
    # 6 input image channel, 6 output channels, 2x2 square convolution
    self.enc_conv1 = nn.Conv2d(6, 12, 2)
    self.enc_conv2 = nn.Conv2d(12, 12, 2)
    self.enc_fc1 = nn.Linear(192, large_hidden)
    self.enc_fc2 = nn.Linear(large_hidden, n_hidden)
    # for decoding
    self.dec_fc1 = nn.Linear(n_hidden, large_hidden)
    self.dec_fc2 = nn.Linear(large_hidden, L*L*6)
    # for abstract functions
    self.h_fc1 = nn.Linear(n_hidden + n_hidden, large_hidden)
    self.h_fc2 = nn.Linear(large_hidden, n_hidden)
    self.v_fc1 = nn.Linear(n_hidden + n_hidden, large_hidden)
    self.v_fc2 = nn.Linear(large_hidden, n_hidden)

    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    self.model_loc = './models/tan_algebra.mdl'

  def channel_last(self, x):
    '''
    batch x channel x L x L => batch x L x L x channel
    useful for visualizing the input, which is given in the LHS form
    '''
    x = x.transpose(1, 2).transpose(2,3).contiguous()
    return x

  def enc(self, x):
    '''
    the x input here is encoded as batch x channel x L x L
    '''
    # Max pooling over a (2, 2) window
    #print (x.size())
    x = F.relu(self.enc_conv1(x))
    # If the size is a square you can only specify a single number
    #print (x.size())
    x = F.relu(self.enc_conv2(x))
    #print (x.size())
    x = x.view(-1, self.num_flat_features(x))
    #print (x.size())
    x = F.relu(self.enc_fc1(x))
    #print (x.size())
    x = F.sigmoid(self.enc_fc2(x))
    #print (x.size())
    return x

  def dec(self, x):
    # here x has dimension [batch x L*L*6]
    x = self.dec_fc1(x)
    x = self.dec_fc2(x)
    # roll it back into [batch x L*L x 6]
    x = x.view(-1, L*L, 6)
    #print (x.size())
    x = F.softmax(x, dim=2)
    # add a smol constant because I am paranoid
    smol_const = to_torch(np.array([1e-6]))
    x = x + smol_const.expand(x.size())
    #print (x.size())
    return x

  def abstr_h(self, x1, x2):
    assert x1.size() == x2.size()
    x1x2 = torch.cat((x1,x2),1)
    x = F.relu(self.h_fc1(x1x2))
    x = F.sigmoid(self.h_fc2(x))
    return x

  def abstr_v(self, x1, x2):
    assert x1.size() == x2.size()
    x1x2 = torch.cat((x1,x2),1)
    x = F.relu(self.v_fc1(x1x2))
    x = F.sigmoid(self.v_fc2(x))
    return x

  def dec_to_board(self, x):
    x = x.view(L,L,6)
    x = x.data.cpu().numpy()
    ret = np.zeros(shape=(L,L))
    for yy in range(L):
      for xx in range(L):
        ret[yy][xx] = np.argmax(x[yy][xx])
    return ret

  def out_to_in(self, x_rec):
    '''
    turns a reconstructed (from a dec operation) kind of data back to its input form
    batch x L*L x channel => batch x channel x L x L
    '''
    x = x_rec.transpose(1,2).contiguous()
    x = x.view(-1, 6, L, L)
    return x


  def auto_enc_cost(self, x_target, x_reconstructed):
    '''
    the target has batch x channel x L x L
    the reconstructed has batch x L * L x channel
    '''
    # move the channel dimention to the last index and flatten it
    x_target = self.channel_last(x_target)
    x_target = x_target.view(-1, 6)
    x_reconstructed = x_reconstructed.view(-1, 6)
    assert x_target.size() == x_reconstructed.size()
    logged_x_rec = torch.log(x_reconstructed)
    cost_value = -torch.sum(x_target * logged_x_rec)
    return cost_value

  def algebra_cost(self, embed_target, embed_constructed):
    diff = embed_target - embed_constructed
    cost_value = torch.sum(torch.pow(diff, 2))
    return cost_value
    
  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

  def enc_dist(self, x, y):
    return torch.sum(torch.pow((x - y), 2))

  # strictly train the embedding batch
  def train_embedding(self, bb):
    self.optimizer.zero_grad()
    b = to_torch(bb)
    # this embedding has shape [20 x 20] = [batch x hidden]
    b_emb = self.enc(b)
    # this reconstruction should have shape [20 x 6 * 6 x 6]
    b_rec = self.dec(b_emb)
    embed_cost = self.auto_enc_cost(b, b_rec)

    cost = embed_cost
    cost.backward()
    self.optimizer.step()
    return cost

  # strictly train the algebra aspect
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

  # attempt to reverse an input x not by encoding but by searching
  def reverse(self, x):
    # FREEEEZER
    for param in self.parameters():
      param.requires_grad = False
    # create 2 candidates
    cand_arg = to_torch(np.random.uniform(low=0.0, high=1.0, size=(1,n_hidden)), True)
    optim = torch.optim.Adam([cand_arg], lr=0.001)
    for i in range(10000):
      optim.zero_grad()
      arg = F.sigmoid(cand_arg)

      return arg

      cost = self.auto_enc_cost(x, self.dec(arg))
      cost.backward()
      optim.step()
      if i % 100 == 0: 
        print (i, cost)
      if cost.data[0] < 0.1:
        print (i, cost)
        return arg

    return arg

  def h_decompose(self, x):
    # FREEEEZER
    for param in self.parameters():
      param.requires_grad = False
    # create 2 candidates
    cand_arg1 = to_torch(np.random.uniform(low=-1.0, high=1.0, size=(1,n_hidden)), True)
    cand_arg2 = to_torch(np.random.uniform(low=-1.0, high=1.0, size=(1,n_hidden)), True)
    optim = torch.optim.Adam([cand_arg1, cand_arg2], lr=0.001)

    for i in range(10000):
      optim.zero_grad()
      arg1 = F.sigmoid(cand_arg1)
      arg2 = F.sigmoid(cand_arg2)
      xyz = self.abstr_h(arg1, arg2)

      rec_cost1 = self.algebra_cost(arg1, self.enc(self.out_to_in(self.dec(arg1))))
      rec_cost2 = self.algebra_cost(arg2, self.enc(self.out_to_in(self.dec(arg2))))
      rec_cost33 = self.algebra_cost(xyz, self.enc(x))
      rec_cost3  = self.auto_enc_cost(x, self.dec(xyz))

      cost = rec_cost1 + rec_cost2 + rec_cost3
      if cost.data[0] < 0.05:
        break
      cost.backward()
      optim.step()
      # cand_arg1.grad.data.zero_()
      # cand_arg2.grad.data.zero_()
      if i % 100 == 0: 
        print (i, rec_cost3)

    print ("reconstruct algebraic cost ")
    print (rec_cost33)
    print ("encoded target")
    print (self.enc(x)[0][:10])
    print ("abstract_h result")
    print (xyz[0][:10])
    print ("encoded decoded abstract_h result")
    print (self.enc(self.out_to_in(self.dec(xyz)))[0][:10])

    print ("generated embedded arg1")
    print (arg1[0][:10])
    print ("enc of dec of generated embedded arg1")
    print (self.enc(self.out_to_in(self.dec(arg1)))[0][:10])

    arg1 = F.sigmoid(cand_arg1)
    arg2 = F.sigmoid(cand_arg2)
    return self.dec(arg1), self.dec(arg2), self.dec(xyz)

if __name__ == '__main__':

  net = ANet().cuda()

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

