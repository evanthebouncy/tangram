from data import *
from embed_mode import *
from algebraic_model import *
from sklearn.manifold import TSNE
from data import *

def visualize_algebra(net):
  # train the algebraic cost
  h_batch, v_batch = gen_train_compose_batch()
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



if __name__ == '__main__':
  net = ANet().cuda()
  model_loc = './models/tan_algebra.mdl'
  net.load_state_dict(torch.load(model_loc))
  # visualize(net)
  # visualize_embedding(net)
  visualize_algebra(net)

