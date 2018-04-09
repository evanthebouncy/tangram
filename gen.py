import numpy as np
from PIL import Image, ImageDraw
import random
from constants import *

# generate and render the tangram 

# each cell can be of 6 states of filled-ness
# empty --- 1 --- 2 --- 3 --- 4 --- full
# ..       xx    x.    .x    xx      xx
# ..       x.    xx    xx    .x      xx

# render constants
RENDER_SCALE = 100

# get the overlap value of two cell content
def get_overlap(cell_contents):
  assert len(cell_contents) == 2, "cyka what are u trying to overlap blin"
  if 0 in cell_contents:
    return sum(cell_contents)
  if 5 in cell_contents:
    return -100
  if 1 in cell_contents and 3 in cell_contents:
    return 5
  if 2 in cell_contents and 4 in cell_contents:
    return 5
  return -100
        
# get the value at a coordinate that is potentially out of bound
def get_value(cell, x, y):
  cell_y, cell_x = cell.shape
  if (0 <= y < cell_y) and (0 <= x < cell_x):
    return cell[y][x]
  else:
    return 0

def h_combine(left_cell, right_cell):
  (left_y, left_x), (right_y, right_x) = left_cell.shape, right_cell.shape
  cells = []
  for shift in reversed(range(left_x + 1)):
    new_y, new_x = max(left_y, right_y), max(left_x, right_x + shift)
    new_cell = np.zeros(shape=(new_y, new_x))
    for yy in range(new_y):
      for xx in range(new_x):
        left_c_y, right_c_y = yy, yy
        left_c_x, right_c_x = xx, xx - shift
        left_value = get_value(left_cell, left_c_x, left_c_y)
        right_value = get_value(right_cell, right_c_x, right_c_y)
        new_cell[yy][xx] = get_overlap([left_value, right_value])
    
    cells.append(new_cell)
    if np.min(new_cell) < 0:
      return cells[-2]
  return cells[-1]

def v_combine(top_cell, bot_cell):
  (top_y, top_x), (bot_y, bot_x) = top_cell.shape, bot_cell.shape
  cells = []
  for shift in reversed(range(top_y + 1)):
    new_y, new_x = max(top_y, bot_y + shift), max(top_x, bot_x)
    new_cell = np.zeros(shape=(new_y, new_x))
    for yy in range(new_y):
      for xx in range(new_x):
        top_c_y, bot_c_y = yy, yy - shift
        top_c_x, bot_c_x = xx, xx
        top_value = get_value(top_cell, top_c_x, top_c_y)
        bot_value = get_value(bot_cell, bot_c_x, bot_c_y)
        new_cell[yy][xx] = get_overlap([top_value, bot_value])

    cells.append(new_cell)
    if np.min(new_cell) < 0:
      return cells[-2]
  return cells[-1]

def gen_rand_board():
  return np.random.randint(6, size=(L, L))

def render_board(board, name="board.png"):

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
    im = Image.new('RGBA', (RENDER_SCALE*L, RENDER_SCALE*L))
    draw = ImageDraw.Draw(im)
    # fill the default color to white
    # draw.polygon([(0,0),(RENDER_SCALE*L,0),(RENDER_SCALE*L,RENDER_SCALE*L),(0,RENDER_SCALE*L)], fill = 'white')
    # draw.rectangle((1, 1, RENDER_SCALE*L-1, RENDER_SCALE*L-1), fill=None, outline='black')
    for c in coords:
      c = RENDER_SCALE * np.array(c)
      draw.polygon([(c[0],c[1]),(c[2],c[3]),(c[4],c[5])], fill = 'black')
    im.save("./renders/{}".format(name), 'PNG')

  coords = board_2_coords(board)
  draw(coords)

def np_to_prim(np_arr):
  assert len(np_arr) == len(SHAPE_TYPES) * len(ORIENTATIONS)
  print ('haha')

# define the pieces
class Piece:

  def __str__(self):
    return str(self.cell)

  def __repr__(self):
    return str(self)

  def to_board(self):
    ret = np.zeros(shape=(L, L))
    for y in range(self.cell.shape[0]):
      for x in range(self.cell.shape[1]):
        ret[y][x] = self.cell[y][x]
    return ret

  def render(self, name):
    render_board(self.to_board(), name)

  def get_construction_str(self):
    '''
    get the construction for this piece as a string
    '''
    if self.p_type not in ['H','V']:
      return "({}_{})".format(self.p_type, self.p_orientation)
    else:
      left = self.p_args[0].get_construction_str()
      right = self.p_args[1].get_construction_str()
      return "({}, {}, {})".format(self.p_type, left, right)

  def piece_to_np(self):
    p_type_np = [0.0, 0.0, 0.0]
    prim_kind_np = [0.0 for _ in range(len(SHAPE_TYPES) * len(ORIENTATIONS))]
    if self.p_type == 'H':
      p_type_np = [0.0, 1.0, 0.0]
    if self.p_type == 'V':
      p_type_np = [0.0, 0.0, 1.0]
    if self.p_type not in ['H', 'V']:
      p_type_np = [1.0, 0.0, 0.0]
      idxx = len(ORIENTATIONS) * SHAPE_TYPES.index(self.p_type) + (self.p_orientation - 1)
      prim_kind_np[idxx] = 1.0

    return np.array(p_type_np), np.array(prim_kind_np)


  def __init__(self, p_type, p_orientation, p_args):
    self.p_type, self.p_orientation, self.p_args = p_type, p_orientation, p_args

    # elementary pieces

    # empty space
    if p_type == '0': self.cell = np.array([[0]])
    # small triangle
    if p_type == '1':
      if p_orientation == 1: self.cell = np.array([[1]])
      if p_orientation == 2: self.cell = np.array([[2]])
      if p_orientation == 3: self.cell = np.array([[3]])
      if p_orientation == 4: self.cell = np.array([[4]])
    # square
    if p_type == '2':
      self.cell = np.array([[5]])
    # medium triangle
    if p_type == '3':
      if p_orientation == 1: self.cell = np.array([[3, 2]])
      if p_orientation == 2: self.cell = np.array([[2],[1]])
      if p_orientation == 3: self.cell = np.array([[4, 1]])
      if p_orientation == 4: self.cell = np.array([[3],[4]])
    # parallelgram
    if p_type == '4':
      if p_orientation == 2: self.cell = np.array([[2],[4]])
      if p_orientation == 1: self.cell = np.array([[3, 1]])
      if p_orientation == 4: self.cell = np.array([[3],[1]])
      if p_orientation == 3: self.cell = np.array([[4, 2]])
    # large triangle
    if p_type == '5':
      if p_orientation == 1: self.cell = np.array([[5, 1],
                                                   [1, 0]])
      if p_orientation == 2: self.cell = np.array([[2, 0],
                                                   [5, 2]])
      if p_orientation == 3: self.cell = np.array([[0, 3],
                                                   [3, 5]])
      if p_orientation == 4: self.cell = np.array([[4, 5],
                                                   [0, 4]])

    # combinators
    # combine 2 pieces horizontally
    if p_type == 'H':
      assert len(p_args) == 2
      left = p_args[0]
      right = p_args[1]
      self.cell = h_combine(left.cell, right.cell)

    # combine 2 pieces vertically
    if p_type == 'V':
      assert len(p_args) == 2
      top = p_args[0]
      down = p_args[1]
      self.cell = v_combine(top.cell, down.cell)

def gen_rand_tangram(shapes):
  rotated = [Piece(x, np.random.randint(1,5), []) for x in shapes]
  work_set = [x for x in rotated]

  while len(work_set) > 1:
    # randomly select 2 elements
    item1 = work_set.pop(random.randint(0,len(work_set)-1))
    item2 = work_set.pop(random.randint(0,len(work_set)-1))

    operator = 'H' if random.random() < 0.5 else 'V'
    item = Piece(operator, 0, [item1, item2])
    work_set.append(item)

  tangram = work_set[0]
  gram_dim = tangram.cell.shape

  if gram_dim[0] < L and gram_dim[1] < L:
    return tangram
  else:
    return gen_rand_tangram(shapes)

def gen_rand_sized_tangram(shapes):
  size_probs = np.exp([float(i) / 2 for i in range(len(shapes))])
  size_probs = size_probs / sum(size_probs)
  size = np.random.choice([i+1 for i in range(len(shapes))], p=size_probs)
  ss = [s for s in shapes]
  random.shuffle(ss)
  chosen_shapes = ss[:size]
  # print(chosen_shapes)
  return gen_rand_tangram(chosen_shapes)

# ========================== testing =====================

# test the render function
def test_render():
  board = gen_rand_board()
  render_board(board)

def test_tree():
  p1 = Piece('1', 1, [])
  print(p1)
  p2 = Piece('3', 4, [])
  print(p2)

  p3 = Piece('H', 0, [p1, p2])
  print(p3)

  p4 = Piece('V', 0, [p3, p1])
  print(p4)

def test_tangram():
  #shapes = ['1', '1', '2', '3', '4', '5', '5'] 
  shapes = ['1','3', '4', '5'] 
  tangram = gen_rand_tangram(shapes)
  print ('type')
  print (tangram.p_type)
  print ('arg1')
  print (tangram.p_args[0])
  print ('arg2')
  print (tangram.p_args[1])
  board = tangram.to_board()
  print(board)
  render_board(board)

  print (tangram.piece_to_np())

if __name__ == "__main__":
  import time
  print("CYKA BLYAT NAXUI")
  # test_render()
  # test_tree()
  test_tangram()


