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

def gen_train_compose():
  tangram = gen_rand_sized_tangram(SHAPES)
  # insist that we do have a non-primitive to be composed
  if tangram.p_type not in ['H', 'V']:
    return gen_train_compose()
  arg1 = tangram.p_args[0]
  arg2 = tangram.p_args[1]
  return tangram.p_type, arg1.to_board(), arg2.to_board(), tangram.to_board()

def gen_train_compose_batch(n=20):
  stuffs = [gen_train_compose() for _ in range(n)]
  h_arg1, h_arg2, h_result, v_arg1, v_arg2, v_result = [],[],[],[],[],[]
  for stuff in stuffs:
    op, arg1, arg2, result = stuff
    arg1, arg2, result = board_to_np(arg1), board_to_np(arg2), board_to_np(result)
    if op == 'H':
      h_arg1.append(arg1)
      h_arg2.append(arg2)
      h_result.append(result)
    else:
      v_arg1.append(arg1)
      v_arg2.append(arg2)
      v_result.append(result)
  return (np.array(h_arg1), np.array(h_arg2), np.array(h_result)),\
         (np.array(v_arg1), np.array(v_arg2), np.array(v_result))

# ==================== tests ===================
def test_np():
  board = gen_rand_sized_tangram(SHAPES).to_board()
  print(board)
  print(board_to_np(board))

def test_gen_train_embed():
  b1s, b2s  = gen_train_embed_batch()
  print(b1s.shape, b2s.shape)

def test_compose():
  op, arg1, arg2, result = gen_train_compose()
  print(op)
  print(arg1)
  print(arg2) 
  print(result)

def test_compose_batch():
  h_train, v_train = gen_train_compose_batch()
  for xx in h_train:
    print(xx.shape)
  for yy in v_train:
    print(yy.shape)

if __name__ == '__main__':
  #test_np()
  #test_gen_train_embed()
  #test_compose()
  test_compose_batch()
