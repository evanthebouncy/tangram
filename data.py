from gen import *

def board_to_np(board):
  ret = np.zeros(shape=(L,L,6))
  for yy in range(L):
    for xx in range(L):
      ret[yy][xx][int(board[yy][xx])] = 1
  return ret

def gen_train_embed():
  # generate a random board
  b = gen_rand_sized_tangram(SHAPES).to_board()
  # with half the chance try to switcharoo
  switch = random.random() < 0.5
  other_b = b if switch else gen_rand_sized_tangram(SHAPES).to_board()
  out = True if switch else False
  return b, other_b, out

def gen_train_embed_batch(n=20):
  bs, other_bs, outs = [], [], []
  for _ in range(n):
    b, other_b, out = gen_train_embed()
    bs.append(board_to_np(b))
    other_bs.append(board_to_np(other_b))
    outs.append([1.0, 0.0] if out else [0.0, 1.0])
  return np.array(bs), np.array(other_bs), np.array(outs)

# ==================== tests ===================
def test_np():
  board = gen_rand_sized_tangram(SHAPES).to_board()
  print(board)
  print(board_to_np(board))

def test_gen_train_embed():
  print(gen_train_embed())

  bs, other_bs, outs = gen_train_embed_batch()
  print(bs.shape, other_bs.shape, outs.shape)

if __name__ == '__main__':
  test_np()
  test_gen_train_embed()
