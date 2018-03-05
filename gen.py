import numpy as np
import Image, ImageDraw

# generate and render the tangram 

# each cell can be of 6 states of filled-ness
# empty --- 1 --- 2 --- 3 --- 4 --- full
# ..       xx    x.    .x    xx      xx
# ..       x.    xx    xx    .x      xx

# some constants
L = 6
RENDER_SCALE = 100

def gen_rand_board():
  return np.random.randint(6, size=(L, L))

def render_board(board):

  def board_2_coords(board):
    coords = []
    for x in range(L):
      for y in range(L):
        # we need to index backwards
        board_cell = board[y][x]
        if board_cell == 0:
          continue
        if board_cell == 1:
          coords.append( [x,y,x+1,y,x,y+1] )
          continue
        if board_cell == 2:
          coords.append( [x,y,x+1,y+1,x,y+1] )
          continue
        if board_cell == 3:
          coords.append( [x+1,y,x+1,y+1,x,y+1] )
          continue
        if board_cell == 4:
          coords.append( [x,y,x+1,y,x+1,y+1] )
          continue
        if board_cell == 5:
          coords.append( [x,y,x+1,y,x,y+1] )
          coords.append( [x+1,y,x+1,y+1,x,y+1] )
          continue
    return coords

  def draw(coords):
    im = Image.new('RGB', (RENDER_SCALE*L, RENDER_SCALE*L))
    draw = ImageDraw.Draw(im)
    # fill the default color to white
    draw.polygon([(0,0),(RENDER_SCALE*L,0),(RENDER_SCALE*L,RENDER_SCALE*L),(0,RENDER_SCALE*L)], fill = 'white')
    for c in coords:
      c = RENDER_SCALE * np.array(c)
      draw.polygon([(c[0],c[1]),(c[2],c[3]),(c[4],c[5])], fill = 'black')
    im.save("./renders/my_pic.png", 'PNG')

  coords = board_2_coords(board)
  draw(coords)

# define the pieces
class Piece:

  def __str__(self):
    return str(self.cell)

  def __init__(self, p_type, p_orientation, p_args):
    self.p_type, self.p_orientation, self.pargs = p_type, p_orientation, p_args

    # elementary pieces

    # empty space
    if p_type == '0': self.cell = np.array([[0]])
    # small triangle
    if p_type == '1':
      if p_orientation == '1': self.cell = np.array([[1]])
      if p_orientation == '2': self.cell = np.array([[2]])
      if p_orientation == '3': self.cell = np.array([[3]])
      if p_orientation == '4': self.cell = np.array([[4]])
    # square
    if p_type == '2':
      self.cell = np.array([[5]])
    # medium triangle
    if p_type == '3':
      if p_orientation == '1': self.cell = np.array([[3, 2]])
      if p_orientation == '2': self.cell = np.array([[2],[1]])
      if p_orientation == '3': self.cell = np.array([[4, 1]])
      if p_orientation == '4': self.cell = np.array([[3],[4]])

    # combinators
    if p_type == 'H':
      assert len(p_args) == 2
      print p_args[0]
      print p_args[1]


# ========================== testing =====================

# test the render function
def test_render():
  board = gen_rand_board()
  render_board(board)

def test_tree():
  p1 = Piece('1', '2', [])
  print p1
  p2 = Piece('3', '2', [])
  print p2

  p3 = Piece('H', '', [p1, p2])
  print p3


if __name__ == "__main__":
  import time
  print "CYKA BLYAT NAXUI"
  test_render()
  test_tree()
#  im = Image.new('RGB', (255, 255))
#  draw = ImageDraw.Draw(im)
#
#  img1 = Image.new('RGBA', (255, 255)) # Use RGBA
#  img2 = Image.new('RGBA', (255, 255)) # Use RGBA
#  draw1 = ImageDraw.Draw(img1)
#  draw2 = ImageDraw.Draw(img2)
#
#  draw1.polygon([(0, 0), (0, 255), (255, 255), (255, 0)], fill = (255,255,255,255))
#
#  transparence = 100 # Define transparency for the triangle.
#  draw2.polygon([(1,1), (20, 100), (100,20)], fill = (200, 0, 0, transparence))
#
#  img = Image.alpha_composite(img1, img2)
#  img.save("my_pic.png", 'PNG')


