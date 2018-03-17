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
  tangram = gen_rand_tangram(SHAPES)
  auto_enc = []
  h_compose = []
  v_compose = []

  def gen_helper(gram):
    auto_enc.append(gram.to_board())
    if gram.p_type == 'H':
      arg1, arg2 = gram.p_args
      h_compose.append((arg1.to_board(), arg2.to_board(), gram.to_board()))
      gen_helper(arg1)
      gen_helper(arg2)
      return
    if gram.p_type == 'V':
      arg1, arg2 = gram.p_args
      v_compose.append((arg1.to_board(), arg2.to_board(), gram.to_board()))
      gen_helper(arg1)
      gen_helper(arg2)
      return
    if gram.p_type not in ['H', 'V']:
      return
    assert 0, "This should never happen cyka blyat"

  gen_helper(tangram)

  return auto_enc, h_compose, v_compose


def gen_train_compose_batch(n=20):
  stuffs = [gen_train_compose() for _ in range(n)]
  embeds = []
  h_arg1, h_arg2, h_result = [],[],[]
  v_arg1, v_arg2, v_result = [],[],[]

  for stuff in stuffs:
    auto_encs, h_composes, v_composes = stuff
    embeds.extend([board_to_np(x) for x in auto_encs])
    for h_ar1, h_ar2, h_res in h_composes:
      h_arg1.append(  board_to_np(h_ar1))
      h_arg2.append(  board_to_np(h_ar2))
      h_result.append(board_to_np(h_res))
    for v_ar1, v_ar2, v_res in v_composes:
      v_arg1.append(  board_to_np(v_ar1))
      v_arg2.append(  board_to_np(v_ar2))
      v_result.append(board_to_np(v_res))

  return np.array(embeds),\
         (np.array(h_arg1), np.array(h_arg2), np.array(h_result)),\
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
  auto, hh, vv = gen_train_compose()
  print(len(auto))
  print(len(hh))
  print(len(vv))

def test_compose_batch():
  embeds, h_train, v_train = gen_train_compose_batch()
  print(embeds.shape)
  for xx in h_train:
    print(xx.shape)
  for yy in v_train:
    print(yy.shape)

if __name__ == '__main__':
  #test_np()
  #test_gen_train_embed()
  #test_compose()
  test_compose_batch()
