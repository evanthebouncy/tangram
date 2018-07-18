import cv2
from gen import *
from data import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from constants import *


def to_torch(x):
  x = Variable(torch.from_numpy(x)).type(torch.cuda.FloatTensor)
  return x

class ENet(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    # for encding
    # 6 input image channel, 6 output channels, 2x2 square convolution
    self.enc_conv1 = nn.Conv2d(6, 12, 2)
    self.enc_conv2 = nn.Conv2d(12, 12, 2)
    self.enc_fc1 = nn.Linear(192, n_hidden)
    self.enc_fc2 = nn.Linear(n_hidden, n_hidden)

    self.dec_fc = nn.Linear(n_hidden, L*L*6)

#    self.meta_param = {
#      'learning_rate' : 0.1,
#    }

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
    x = self.dec_fc(x)
    # roll it back into [batch x L*L x 6]
    x = x.view(-1, L*L, 6)
    #print (x.size())
    # add a smol constant because I am paranoid
    smol_const = to_torch(np.array([1e-6]))
    x = x + smol_const.expand(x.size())

    x = F.softmax(x, dim=2)
    #print (x.size())
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
    
#  def forward(self, x):
#    e_x = self.enc(x)
#    return e_x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

  def enc_dist(self, x, y):
    return torch.sum(torch.pow((x - y), 2))

if __name__ == '__main__':

  net = Net().cuda()
  #print(net)

  optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
  model_loc = './models/tan_embed.mdl'

  for i in range(100000):
    optimizer.zero_grad()
    b = gen_train_embed_batch()
    b = to_torch(b)
    # this embedding has shape [20 x 20] = [batch x hidden]
    b_emb = net.enc(b)
    # this reconstruction should have shape [20 x 6 * 6 x 6]
    b_rec = net.dec(b_emb)

    cost = net.auto_enc_cost(b, b_rec)
    cost.backward()
    optimizer.step()

    if i % 1000 == 0:
      print ("===== d i a g n o s t i c    a e s t h e t i c s =====")
      print("cost ", cost)
      for jjj in range(20):
        orig_board = net.dec_to_board(net.channel_last(b)[jjj])
        rec_board = net.dec_to_board(b_rec[jjj])
        # print (orig_board)
        # print (rec_board)
        render_board(orig_board, "board_{}_orig.png".format(jjj))
        render_board(rec_board, "board_{}_rec.png".format(jjj))

        torch.save(net.state_dict(), model_loc) 


