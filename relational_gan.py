from gen import *
from data import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from constants import *
from utils import *

'''
overall scheme : train a GAN that work as follows

1) the generator that takes y => x1 x2 such that x1 and x2 forms y

2) the discriminator that takes x1 x2 y and tell if it is real or not
'''

# =================================== NETWORK MODELS ===================================
# the backward function that decompose shapes
class Gen(nn.Module):

  def __init__(self):
    super(Gen, self).__init__()

    # for encding
    # 6 input image channel, 6 output channels, 2x2 square convolution
    self.enc_conv1 = nn.Conv2d(6, 12, 2)
    self.enc_conv2 = nn.Conv2d(12, 12, 2)
    self.enc_fc1 = nn.Linear(192, large_hidden)
    self.enc_fc2 = nn.Linear(large_hidden, n_hidden)

    # for decoding
    self.dec_fc1 = nn.Linear(n_hidden, large_hidden)
    self.dec_fc2 = nn.Linear(large_hidden, L*L*6)

    # for inverting (h and v)
    self.inv_h_fc1 = nn.Linear(n_hidden, large_hidden)
    self.inv_h_fc2 = nn.Linear(large_hidden, 2 * n_hidden)
    self.inv_v_fc1 = nn.Linear(n_hidden, large_hidden)
    self.inv_v_fc2 = nn.Linear(large_hidden, 2 * n_hidden)

    lr = 0.0002
    self.opt = torch.optim.Adam(self.parameters(), lr=lr)

  def enc(self, x):
    '''
    the x input here is encoded as batch x channel x L x L
    encode into a n_hidden latent representation
    '''
    x = F.relu(self.enc_conv1(x))
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

  def invert_latent(self, op_type, e):
    '''
    takes in a latent vector e
    takes in the inversion type op_type (H or V)
    outputs 2 embeddings e1 and e2 that forms e
    '''

    inv_map = {
        'H' : {
            'inv_fc1' : self.inv_h_fc1,
            'inv_fc2' : self.inv_h_fc2,
          },
        'V' : {
            'inv_fc1' : self.inv_v_fc1,
            'inv_fc2' : self.inv_v_fc2,
          }
        }
    
    # we first enlarge the encoding to a large hidden dimension to get ready
    e_large = F.relu(inv_map[op_type]['inv_fc1'](e))
    e1e2 = F.relu(inv_map[op_type]['inv_fc2'](e_large))
    e1, e2 = torch.split(e1e2, n_hidden, dim=1)
    return e1, e2

  def forward(self, op_type, x):
    e = self.enc(x)
    e1, e2 = self.invert_latent(op_type, e)
    x1, x2 = self.dec(e1), self.dec(e2)
    return x1, x2

# the relational discriminator
class Dis(nn.Module):

  def __init__(self):
    super(Dis, self).__init__()

    # for encding
    # 6 input image channel, 6 output channels, 2x2 square convolution
    self.enc_conv1 = nn.Conv2d(6, 12, 2)
    self.enc_conv2 = nn.Conv2d(12, 12, 2)
    self.enc_fc1 = nn.Linear(192, large_hidden)
    self.enc_fc2 = nn.Linear(large_hidden, n_hidden)

    # for discriminating
    self.discrim_h = nn.Linear(n_hidden * 3, 1)
    self.discrim_v = nn.Linear(n_hidden * 3, 1)

    lr = 0.0002
    self.opt = torch.optim.Adam(self.parameters(), lr=lr)

  def enc(self, x):
    '''
    the x input here is encoded as batch x channel x L x L
    encode into a n_hidden latent representation
    '''
    x = F.relu(self.enc_conv1(x))
    x = F.relu(self.enc_conv2(x))
    x = x.view(-1, num_flat_features(x))
    x = F.relu(self.enc_fc1(x))
    e = F.sigmoid(self.enc_fc2(x))
    return e

  def forward(self, op_type, x, x1, x2):
    dis_map = {
        'H' : self.discrim_h,
        'V' : self.discrim_v,
        }
    e, e1, e2 = self.enc(x), self.enc(x1), self.enc(x2)
    ee1e2 = torch.cat( (e,e1,e2), 1 )
    return F.sigmoid(dis_map[op_type](ee1e2))

# draws the first of the batch from these fools
def draw_stuff(op_type, x, x1, x2, x1_rec, x2_rec):
  x_board =      dec_to_board(channel_last(x)[0])  , op_type + "_x.png"
  x1_board =     dec_to_board(channel_last(x1)[0]) , op_type + "_x1.png"
  x2_board =     dec_to_board(channel_last(x2)[0]) , op_type + "_x2.png"
  x1_rec_board = dec_to_board(x1_rec[0])           , op_type + "_x1_r.png"
  x2_rec_board = dec_to_board(x2_rec[0])           , op_type + "_x2_r.png"

  render_board(*x_board)
  render_board(*x1_board)
  render_board(*x2_board)
  render_board(*x1_rec_board)
  render_board(*x2_rec_board)

# ======================= TRAINING PROCEDURES =========================
criterion = nn.BCELoss()

def train_generator(gen, dis_score):
  real_labels = Variable(torch.ones(dis_score.size()).cuda())
  gen.zero_grad()
  g_loss = criterion(dis_score, real_labels)
  g_loss.backward()
  gen.opt.step()
  return g_loss

def train_discriminator(dis, real_tuples, fake_tuples, m):
  real_labels = Variable(torch.ones( (m,1)).cuda())
  fake_labels = Variable(torch.zeros((m,1)).cuda())

  dis.zero_grad()

  outputs = dis(*real_tuples)
  real_loss = criterion(outputs, real_labels)
  real_score = outputs
  
  outputs = dis(*fake_tuples)
  fake_loss = criterion(outputs, fake_labels)
  fake_score = outputs

  d_loss = real_loss + fake_loss
  d_loss.backward()
  dis.opt.step()
  return d_loss, real_score, fake_score

if __name__ == '__main__':
  gen = Gen().cuda()
  dis = Dis().cuda()

  for i in range(10000):
    # step 0 : sample a batch of decompositions
    a_batch = gen_train_planner_batch(len(SHAPES))
    bb, aa, h_batch, v_batch = a_batch

    # ------------------- train the H decomposition -------------------
    h1, h2, h = h_batch
    h1, h2, h = to_torch(h1), to_torch(h2), to_torch(h)

    # step 1 : sample a reconstruction from generator
    h1_rec, h2_rec =  gen('H', h)
    # step 1.1 : flip the channel so they can be inputted agian
    h1_rec_ch, h2_rec_ch = out_to_in(h1_rec), out_to_in(h2_rec)

    # step 2 : get the score of the discriminator
    dis_score = dis('H', h, h1_rec_ch, h2_rec_ch) 

    # step 3 : train the generator on the discrim score
    g_loss = train_generator(gen, dis_score)

    # step 4 : train the discriminator
    # step 4.1 : re-sample a reconstruction from generator
    h1_rec, h2_rec =  gen('H', h)
    h1_rec_ch, h2_rec_ch = out_to_in(h1_rec), out_to_in(h2_rec)
    # step 4.2 : construct the real and fake tuples and train
    tup_real = ('H', h, h1, h2) 
    tup_fake = ('H', h, h1_rec_ch, h2_rec_ch)
    m = len(h)
    d_loss, real_score, fake_score = train_discriminator(dis, tup_real, tup_fake, m)

    # print some statistics
    if i % 200 == 0:
      print (i, g_loss, d_loss)
      draw_stuff("H", h, h1, h2, h1_rec, h2_rec)

#     print (h1_rec.size())
#     h1_rec_ch = out_to_in(h1_rec)
#     h2_rec_ch = out_to_in(h2_rec)
# 
#     dis2 = dis('H', h, h1_rec_ch, h2_rec_ch)
# 
# 
#     print (dis1, dis2)
# 
#     tup_real = ('H', h, h1, h2) 
#     tup_fake = ('H', h, h1_rec_ch, h2_rec_ch)
#     m = len(h)
# 
#     print (train_discriminator(dis, tup_real, tup_fake, m)[0])
# 
# 
#   assert 0, "asdf"



