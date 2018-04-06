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

# this cost is just the diff of the norm
def diff_norm_cost(e_target, e):
  diff = e_target - e
  cost_value = torch.sum(diff.norm(dim=1))
  return cost_value

# simple cross entropy cost (might be numerically unstable if pred has 0)
def xentropy_cost(x_target, x_pred):
  assert x_target.size() == x_pred.size(), \
      "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
  logged_x_pred = torch.log(x_pred)
  cost_value = -torch.sum(x_target * logged_x_pred)
  return cost_value

# the decoding cost on the picture level, with channel swaps
def dec_cost(x_target, x_reconstructed):
  '''
  the target has batch x channel x L x L
  the reconstructed has batch x L * L x channel
  '''
  # move the channel dimention to the last index and flatten it
  x_target = channel_last(x_target)
  x_target = x_target.view(-1, 6)
  x_reconstructed = x_reconstructed.view(-1, 6)
  return xentropy_cost(x_target, x_reconstructed)


