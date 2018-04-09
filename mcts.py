from gen import *
from data import *

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from constants import *
from utils import *

from gauss_planner import *

class MCTS:

  def __init__(self, pnet):
    print ("L O A D E D")
    self.pnet = pnet

  def search(self, board):
    b = to_torch(np.array([board_to_np(board)]))

    e = self.pnet.enc(b)
    return self.recursive_search(e)

  def recursive_search(self, e):
    op, e1, e2 = self.pnet.sample_decompose(e)
    if op not in ['H', 'V']:
      p_type, orientation = op
      to_ret = Piece(p_type, orientation, [])
      #to_ret.render(to_ret.get_construction_str()+".png")
      return to_ret, (e, op)
    else:
      part1, construction1 = self.recursive_search(e1)
      part2, construction2 = self.recursive_search(e2)
      to_ret = Piece(op, 0, [part1, part2])
      #to_ret.render(to_ret.get_construction_str()+".png")
      return to_ret, (e, op, construction1, construction2)

def test_mcts1(mcts):
  gram_subset = np.random.choice(SHAPES, size=2, replace=False)
  # gram_subset = np.random.choice(['1', '3'], size=2, replace=False)
  tangram = gen_rand_tangram(gram_subset)
  gram_board = tangram.to_board()

  gram_left, gram_right = tangram.p_args
  gram_left, gram_right = gram_left.to_board(), gram_right.to_board()
  render_board(gram_left,  "tangram_left.png")
  render_board(gram_right,  "tangram_right.png")

  render_board(gram_board,  "tangram_input.png")

  for i in range(1):
    print (" working ", i)
    result, construction = mcts.search(gram_board)
    print (construction)
    result_board = result.to_board()
    render_board(result_board, "tangram_result_{}.png".format(i))
    if tangram.is_same(result):
      print ("SOLVED! ", i)
      break


  # print ("left ")
  # print (mcts.pnet.enc(to_torch(np.array([board_to_np(gram_left)]))))
  # print ("right ")
  # print (mcts.pnet.enc(to_torch(np.array([board_to_np(gram_right)]))))
  # b = to_torch(np.array([board_to_np(gram_board)]))
  # e = mcts.pnet.enc(b)
  # op, e1, e2 = mcts.pnet.sample_decompose(e)
  # print ("decomposed left")
  # print (e1)
  # print ("decomposed right")
  # print (e2)


if __name__ == "__main__":
  pnet = GNet().cuda()
  model_loc = pnet.model_loc
  pnet.load_state_dict(torch.load(model_loc))
  mcts = MCTS(pnet)

  test_mcts1(mcts)
