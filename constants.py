# all the constant globals shared across codes
OPS = ['P', 'H', 'V']
n_hidden = 40
# n_hidden = 120 # good embedding bad decomposition
large_hidden = 6 * n_hidden
L = 6
SHAPE_TYPES = ['1', '2', '3', '4', '5']
SHAPES = ['1', '1', '2', '3', '4', '5', '5'] 
ORIENTATIONS = [1, 2, 3, 4]

SXO = [(s,o) for s in SHAPE_TYPES for o in ORIENTATIONS]
ACTIONS = ['H', 'V'] + SXO
