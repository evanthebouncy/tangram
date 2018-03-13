from data import *
#from embed_mode import *
from algebraic_model import *
from sklearn.manifold import TSNE

if __name__ == '__main__':
  net = Net().cuda()
  model_loc = './models/tan_algebra.mdl'
  net.load_state_dict(torch.load(model_loc))

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

