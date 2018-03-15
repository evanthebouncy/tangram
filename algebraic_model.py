from gen import *
from data import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from constants import *

large_hidden = 4 * n_hidden

def to_torch(x):
  x = Variable(torch.from_numpy(x)).type(torch.cuda.FloatTensor)
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
    
    # for book keeping
    self.enc_dec_weights = [
      self.enc_conv1,
      self.enc_conv2,
      self.enc_fc1,
      self.enc_fc2,
      self.dec_fc1,
      self.dec_fc2,
    ]

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
  def train_algebra(self, hv_batch, freeze=True):
    self.optimizer.zero_grad()
    # train the algebraic cost
    h_batch, v_batch = hv_batch
    # H operator
    h_arg1, h_arg2, h_result = h_batch
    h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
    h_pred = self.abstr_h(self.enc(h_arg1), self.enc(h_arg2))
    h_pred_dec = self.dec(h_pred)
    h_cost = self.auto_enc_cost(h_result, h_pred_dec)

    h_emb_cost = self.algebra_cost(self.enc(h_result), h_pred)

    # V operator
    v_arg1, v_arg2, v_result = v_batch
    v_arg1, v_arg2, v_result = to_torch(v_arg1), to_torch(v_arg2), to_torch(v_result)
    v_pred = self.abstr_v(self.enc(v_arg1), self.enc(v_arg2))
    v_pred_dec = self.dec(v_pred)
    v_cost = self.auto_enc_cost(v_result, v_pred_dec)

    v_emb_cost = self.algebra_cost(self.enc(v_result), v_pred)

    rec_cost = h_cost + v_cost
    emb_cost = h_emb_cost + v_emb_cost

    cost = rec_cost + emb_cost
    cost.backward()

    if freeze:
      # freeze these weights
      for param in self.enc_dec_weights:
        param.zero_grad()
      self.optimizer.step()

    return rec_cost, emb_cost

if __name__ == '__main__':

  net = ANet().cuda()

  # train the embeddings for some iterations
  for i in range(10001):

    # train for the embedding cost
    cost = net.train_embedding(gen_train_embed_batch())

    if i % 1000 == 0:
      print ("===== e m b e d d i n g    a e s t h e t i c s ===== ", i)
      print("cost ", cost)
      torch.save(net.state_dict(), net.model_loc) 
      b = gen_train_embed_batch()
      b = to_torch(b)
      b_rec = net.dec(net.enc(b))
      for jjj in range(20):
        orig_board = net.dec_to_board(net.channel_last(b)[jjj])
        rec_board = net.dec_to_board(b_rec[jjj])
        render_board(orig_board, "embed_board_{}_orig.png".format(jjj))
        render_board(rec_board, "embed_board_{}_rec.png".format(jjj))

  # train the algebra for some iterations
  for i in range(10001):
    if i < 4000:
      # freeze the weight, trian the algebra hard
      alg_cost = net.train_algebra(gen_train_compose_batch())
    else:
      # jointly train both the algebra and the embedding
      alg_cost = net.train_algebra(gen_train_compose_batch(), freeze=False)
#      emb_cost = net.train_embedding(gen_train_embed_batch())

    if i % 1000 == 0:
      print ("===== a l g e b r a i c    a e s t h e t i c s ===== ", i)
      print("cost ", alg_cost)
      torch.save(net.state_dict(), net.model_loc) 

      h_batch, v_batch = gen_train_compose_batch()
      # H operator
      h_arg1, h_arg2, h_result = h_batch
      h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
      h_pred = net.abstr_h(net.enc(h_arg1), net.enc(h_arg2))
      b_rec = net.dec(h_pred)

      for jjj in range(len(h_arg1)):
        orig_board = net.dec_to_board(net.channel_last(h_result)[jjj])
        arg1_board = net.dec_to_board(net.channel_last(h_arg1)[jjj])
        arg2_board = net.dec_to_board(net.channel_last(h_arg2)[jjj])
        rec_board = net.dec_to_board(b_rec[jjj])
        render_board(arg1_board, "algebra_board_{}_arg1.png".format(jjj))
        render_board(arg2_board, "algebra_board_{}_arg2.png".format(jjj))
        render_board(orig_board, "algebra_board_{}_result.png".format(jjj))
        render_board(rec_board,  "algebra_board_{}_predict.png".format(jjj))

# diagnosis
#    max_element = 0.0
#    max_grad = 0.0
#    for param in net.parameters():
#      try:
#        x = param.data.cpu().numpy()
#        x_grad = param.grad.data.cpu().numpy()
#        max_element = max(max_element, np.max(np.abs(x)))
#        max_grad = max(max_grad, np.max(np.abs(x_grad)))
#      except: pass
#    print ("cost {} max weight {} max grad {}".format(cost, max_element, max_grad))



  # then train the algebra



#    # train the algebraic cost
#    h_batch, v_batch = gen_train_compose_batch()
#    # H operator
#    h_arg1, h_arg2, h_result = h_batch
#    h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
#    h_pred = net.abstr_h(net.enc(h_arg1), net.enc(h_arg2))
#    h_result_enc = net.enc(h_result)
#    h_cost = net.algebra_cost(h_result_enc, h_pred)
#
#    # V operator
#    v_arg1, v_arg2, v_result = v_batch
#    v_arg1, v_arg2, v_result = to_torch(v_arg1), to_torch(v_arg2), to_torch(v_result)
#    v_pred = net.abstr_v(net.enc(v_arg1), net.enc(v_arg2))
#    v_result_enc = net.enc(v_result)
#    v_cost = net.algebra_cost(v_result_enc, v_pred)
#
#    alg_cost = h_cost + v_cost
#
#    # put another auto-encoding cost for safety ...? decode both h_emb and v_emb
#    h_pred_dec = net.dec(h_pred)
#    h_embed_cost = net.auto_enc_cost(h_result, h_pred_dec)
#    v_pred_dec = net.dec(v_pred)
#    v_embed_cost = net.auto_enc_cost(v_result, v_pred_dec)
#    hv_embed_cost = 0.1 * (h_embed_cost + v_embed_cost)


