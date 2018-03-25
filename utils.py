import torch
from torch.autograd import Variable
def to_torch(x, req = False):
  x = Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), requires_grad = req)
  return x

def channel_last(x):
  '''
  batch x channel x L x L => batch x L x L x channel
  useful for visualizing the input, which is given in the LHS form
  '''
  x = x.transpose(1, 2).transpose(2,3).contiguous()
  return x

def num_flat_features(x):
  size = x.size()[1:]  # all dimensions except the batch dimension
  num_features = 1
  for s in size:
    num_features *= s
  return num_features
