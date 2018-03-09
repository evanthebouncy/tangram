from gen import *

# put it into a numpy shape, and flip it so it is channel x L x L
def board_to_np(board):
  ret = np.zeros(shape=(L,L,6))
  for yy in range(L):
    for xx in range(L):
      ret[yy][xx][int(board[yy][xx])] = 1
  return np.transpose(ret, (2,0,1))

def gen_train_embed_batch(n=20):
  return np.array([board_to_np(gen_rand_sized_tangram(SHAPES).to_board())\
                   for _ in range(n)])
  #         np.array([board_to_np(gen_rand_sized_tangram(SHAPES).to_board())\
  #                   for _ in range(n)])
# ==================== tests ===================
def test_np():
  board = gen_rand_sized_tangram(SHAPES).to_board()
  print(board)
  print(board_to_np(board))

def test_gen_train_embed():
  b1s, b2s  = gen_train_embed_batch()
  print(b1s.shape, b2s.shape)

if __name__ == '__main__':
  test_np()
  #test_gen_train_embed()
