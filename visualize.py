from data import *
from embed_mode import *
from algebraic_model import *
from sklearn.manifold import TSNE
from data import *

def visualize_interpolation(net, e1, e2):
  diff = e2 - e1
  delta = diff / 20
  for i in range(20 + 1):
    e_middle = e1 + delta * i
    render_board(net.dec_to_board(net.dec(e_middle)), "interpo_{}.png".format(i))

def visualize_distr(net):
  for i in range(20):
    cand_arg = to_torch(np.random.uniform(low=-1.0, high=1.0, size=(1,n_hidden)))
    arg = F.sigmoid(cand_arg)
    render_board(net.dec_to_board(net.dec(arg)), "distr_sample_{}.png".format(i))

def visualize_algebra(net):
  # train the algebraic cost
  _, h_batch, v_batch = gen_train_compose_batch()
  # H operator
  h_arg1, h_arg2, h_result = h_batch
  h_arg1, h_arg2, h_result = to_torch(h_arg1), to_torch(h_arg2), to_torch(h_result)
  h_pred = net.abstr_h(net.enc(h_arg1), net.enc(h_arg2))
  b_rec = net.dec(h_pred)

  for jjj in range(len(h_arg1)):
    orig_board = net.dec_to_board(net.channel_last(h_result)[jjj])
    arg1_board = net.dec_to_board(net.channel_last(h_arg1)[jjj])
    arg2_board = net.dec_to_board(net.channel_last(h_arg2)[jjj])
    rec_board = net.dec_to_board(b_rec[jjj])
    print (orig_board)
    print (rec_board)
    render_board(arg1_board, "board_{}_arg1.png".format(jjj))
    render_board(arg2_board, "board_{}_arg2.png".format(jjj))
    render_board(orig_board, "board_{}_truth.png".format(jjj))
    render_board(rec_board, "board_{}_predict.png".format(jjj))

def visualize_embedding(net):
  # generate 100 random tangrams
  b = gen_train_embed_batch(500)
  b = to_torch(b)    
  b_emb = net.enc(b) 

  print (b_emb)

  X = b_emb.data.cpu().numpy()
  X_embedded = TSNE(n_components=2).fit_transform(X)

  print(X_embedded)

  boards = net.channel_last(b)
  boards = [net.dec_to_board(bb) for bb in boards]

  dicc = dict()

  for ii, board in enumerate(boards):
    print(board)
    render_board(board, "vis_{}.png".format(ii))
    dicc['vis_{}.png'.format(ii)] = list(X_embedded[ii])

  with open('data_vis.js', "w") as fd:
    fd.write('data_vis = ' + repr(dicc))

def test_robust(net):
  b = gen_train_embed_batch(1)
  b = to_torch(b)    
  b_emb = net.enc(b) 
  # randomly perturb it
  nearbys = [to_torch(np.random.uniform(low=-0.1, high=0.1, size=(1,n_hidden)), False) for _ in range(20)]
  nearbys = [b_emb + nbys for nbys in nearbys]

  render_board(net.dec_to_board(net.channel_last(b)[0]), "orig.png")
  for ii, nby in enumerate(nearbys):
    render_board(net.dec_to_board(net.dec(nby)), "nearby_{}.png".format(ii))

def test_modality(net):
  b = gen_train_embed_batch(1)
  b = to_torch(b)    
  b_emb = net.enc(b) 
  b_alt = net.reverse(b)

  print (b_emb[0][:10])
  print (b_alt[0][:10])
  print (net.algebra_cost(b_emb, b_alt))

  render_board(net.dec_to_board(net.channel_last(b)[0]), "orig.png")
  render_board(net.dec_to_board(net.dec(b_emb)), "emb_render.png")
  render_board(net.dec_to_board(net.dec(b_alt)), "alt_render.png")
  visualize_interpolation(net, b_alt, b_emb)

def test_1step_dec(anet, b, xxx):
  b = to_torch(b[1:2, :])
  print ("did we do it boys?")
  b_board    = anet.dec_to_board(anet.channel_last(b)[0])
  render_board(b_board,    "{}_decompose_target.png".format(xxx))

  arg1, arg2, rec = anet.h_decompose(b)

  print (arg1.size())
  print (arg2.size())
  print (rec.size())

  arg1_board = anet.dec_to_board(arg1[0])
  arg2_board = anet.dec_to_board(arg2[0])
  rec_board = anet.dec_to_board(rec[0])
  render_board(arg1_board, "{}_decompose_arg1.png".format(xxx))
  render_board(arg2_board, "{}_decompose_arg2.png".format(xxx))
  render_board(rec_board,  "{}_decompose_rec.png".format(xxx))

if __name__ == '__main__':
  net = ANet().cuda()
  model_loc = './models/tan_algebra.mdl'
  net.load_state_dict(torch.load(model_loc))
  # visualize(net)
  # visualize_distr(net)
  # visualize_embedding(net)
  visualize_algebra(net)
  # test_robust(net)
  #test_modality(net)

