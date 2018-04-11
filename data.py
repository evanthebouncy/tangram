from gen import *

# put it into a numpy shape, and flip it so it is channel x L x L
def board_to_np(board):
  ret = np.zeros(shape=(L,L,6))
  for yy in range(L):
    for xx in range(L):
      ret[yy][xx][int(board[yy][xx])] = 1
  return np.transpose(ret, (2,0,1))

# put into numpy shape the actions
def action_to_np(act):
  ret = np.zeros(shape=len(ACTIONS))
  ret[ACTIONS.index(act)] = 1.0
  return ret

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

def gen_train_planner(size, tangram = None):
  gram_subset = np.random.choice(SHAPES, size, replace=False)
  tangram = gen_rand_tangram(gram_subset) if tangram is None else tangram
  actions = []
  h_compose = []
  v_compose = []

  def gen_helper(gram):
    if gram.p_type == 'H':
      arg1, arg2 = gram.p_args
      h_compose.append((arg1.to_board(), arg2.to_board(), gram.to_board()))
      gen_helper(arg1)
      gen_helper(arg2)
      actions.append((gram.to_board(), 'H'))
      return
    if gram.p_type == 'V':
      arg1, arg2 = gram.p_args
      v_compose.append((arg1.to_board(), arg2.to_board(), gram.to_board()))
      gen_helper(arg1)
      gen_helper(arg2)
      actions.append((gram.to_board(), 'V'))
      return
    if gram.p_type not in ['H', 'V']:
      actions.append((gram.to_board(), (gram.p_type, gram.p_orientation)))
      return
    assert 0, "This should never happen cyka blyat"

  gen_helper(tangram)

  return actions, h_compose, v_compose

def gen_train_planner_batch(size, n=20, stuffs=[]):
  stuffs = [gen_train_planner(size) for _ in range(n)] if stuffs == [] else stuffs
  embeds = []
  actions = []
  h_arg1, h_arg2, h_result = [],[],[]
  v_arg1, v_arg2, v_result = [],[],[]

  for stuff in stuffs:
    input_actions, h_composes, v_composes = stuff
    embeds.extend([board_to_np(x[0]) for x in input_actions])
    actions.extend([action_to_np(x[1]) for x in input_actions])
    for h_ar1, h_ar2, h_res in h_composes:
      h_arg1.append(  board_to_np(h_ar1))
      h_arg2.append(  board_to_np(h_ar2))
      h_result.append(board_to_np(h_res))
    for v_ar1, v_ar2, v_res in v_composes:
      v_arg1.append(  board_to_np(v_ar1))
      v_arg2.append(  board_to_np(v_ar2))
      v_result.append(board_to_np(v_res))


  return np.array(embeds), np.array(actions),\
         (np.array(h_arg1), np.array(h_arg2), np.array(h_result)),\
         (np.array(v_arg1), np.array(v_arg2), np.array(v_result))

def self_supervise(tangram, agent):
  '''
  given a tangram, if it is solvable by the agent, use agent's proposal as supervision
  if not, use just enough supervision on top level
  '''
  def _self_supervise(gram):
    board = gram.to_board()
    succ, result = agent.n_search(board)
    # return the result suggested by the agent if the agent has an idea
    if succ:
      # print('succ')
      return result
    # otherwise, recursively tries to still use the agent as much as possible
    else:
      # print('fail')
      # agent you fucking debil fuck up the primitive, oh well
      if gram.p_args == []:
        return gram
      else:
        op = gram.p_type
        arg1, arg2 = gram.p_args
        arg1_result = _self_supervise(arg1)
        arg2_result = _self_supervise(arg2)
        return Piece(op, 0, [arg1_result, arg2_result]) 
  return [_self_supervise(tangram)]

#   board = tangram.to_board()
#   succ, result = agent.n_search(board)
#   if succ:
#     return [result]
#   else:
#     # return [_self_supervise(tangram), _self_supervise(result)]
#     return [_self_supervise(tangram)]
  
  # return _self_supervise(tangram)

def gen_train_RL(size, agent):
  gram_subset = np.random.choice(SHAPES, size, replace=False)
  tangram = gen_rand_tangram(gram_subset)
  tangram_recs = self_supervise(tangram, agent)
  return [gen_train_planner(0, tangram = tangram_rec) for tangram_rec in tangram_recs]

def gen_train_RL_batch(size, agent, n=20):
  # super hack joining list of list O_O
  stuffs = sum([gen_train_RL(size, agent) for _ in range(n)],[])
  return gen_train_planner_batch(0, stuffs=stuffs)

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
  actions, embeds, h_train, v_train = gen_train_compose_batch()
  print(embeds.shape)
  for xx in h_train:
    print(xx.shape)
  for yy in v_train:
    print(yy.shape)

def test_planner():
  action, hh, vv = gen_train_planner()
  print(action)
  print(hh)
  print(vv)
  print(len(action))
  print(len(hh))
  print(len(vv))

def test_planner_batch():
  embeds, actions, h_train, v_train = gen_train_planner_batch()
  print(embeds.shape)
  print(actions.shape)
  for xx in h_train:
    print(xx.shape)
  for yy in v_train:
    print(yy.shape)


if __name__ == '__main__':
  #test_np()
  #test_gen_train_embed()
  #test_compose()
  #test_compose_batch()
  #test_planner()
  test_planner_batch()
