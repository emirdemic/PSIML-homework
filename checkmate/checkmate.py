import numpy as np 
from PIL import Image
import os 


def load_images(path):

	tiles = {
		'black' : None,
		'white' : None
	}

	black_pieces = {
		'b' : None,
		'k' : None,
		'n' : None,
		'p' : None,
		'q' : None,
		'r' : None
	}

	white_pieces = {
		'B' : None,
		'K' : None,
		'N' : None,
		'P' : None,
		'Q' : None,
		'R' : None
	}

	for file in os.listdir(os.path.join(path + r'\pieces' + r'\black')):
		if file.lower().startswith('knight'):
			black_pieces['n'] = Image.open(path + r'\pieces' + r'\black' + '\\' + file).convert('LA')
		else: 
			black_pieces[file[0]] = Image.open(path + r'\pieces' + r'\black' + '\\' + file).convert('LA')

	for file in os.listdir(os.path.join(path + r'\pieces' + r'\white')):
		if file.lower().startswith('knight'):
			white_pieces['N'] = Image.open(path + r'\pieces' + r'\white' + '\\' + file).convert('LA')
		else:
			white_pieces[file[0].upper()] = Image.open(path + r'\pieces' + r'\white' + '\\' + file).convert('LA')
	for file in os.listdir(os.path.join(path + r'\tiles')):
		tiles[file[:-4].lower()] = Image.open(path + r'\tiles' + '\\' + file).convert('LA')

	return tiles, white_pieces, black_pieces


def merge_images(tile, piece):

	combined = tile.copy()
	combined.paste(
		piece,
		mask = piece.split()[1]
		)

	return combined


def combine_images(tiles, white_pieces, black_pieces):

	combined_dictionary = {}
	for tile in tiles:
		for piece in white_pieces:
			combined_dictionary[tile + '_' + 'w' + '_' + piece] = merge_images(tiles[tile], white_pieces[piece])

	for tile in tiles:
		for piece in black_pieces:
			combined_dictionary[tile + '_' + 'b' + '_' + piece] = merge_images(tiles[tile], black_pieces[piece])

	return combined_dictionary



def get_tile_size(chessboard, first_x, first_y):

	i_color = chessboard[first_x, first_y]
	tile_size = 0
	for i in range(first_x, np.shape(chessboard)[0]):
		if np.all(chessboard[i, first_y] != i_color):
			break
		else:
			tile_size += 1

	return tile_size



def extract_chessboard(chessboard, first_x, first_y, tile_size):

	chessboard = chessboard[first_x : first_x + (tile_size * 8), first_y : first_y + (tile_size * 8)]
	
	return chessboard


def rescale_images(tile_size, images):
	for image in images:
		images[image] = np.array(images[image].resize((tile_size, tile_size), resample = Image.BILINEAR))

	return images


def get_tile_colors(black_tile, white_tile):
	black_tile_color = black_tile[0, 0]
	white_tile_color = white_tile[0, 0]

	return black_tile_color, white_tile_color


def get_fen(chessboard, rescaled_images, tile_size, black_tile_color, white_tile_color):
	chessboard_matrix = ''
	empty_tiles = 0
	fen_notation = ''
	row_index = 0
	for row in range(8):
		column_index = 0
		for column in range(8):
			tile = chessboard[0+row_index : tile_size+row_index, 0+column_index : tile_size+column_index]
			if np.all(tile == black_tile_color) or np.all(tile == white_tile_color):
				empty_tiles += 1
				chessboard_matrix += '*'
			else:
				if empty_tiles > 0:
					fen_notation += str(empty_tiles)
					empty_tiles = 0
				flattened_tile = np.ndarray.flatten(tile)
				corr_score = 0
				chess_piece = ''
				for image in rescaled_images:
					flattened_image = np.ndarray.flatten(rescaled_images[image])
					corr_coef = np.corrcoef(flattened_tile, flattened_image)[0, 1]
					if corr_coef > corr_score:
						corr_score = corr_coef 
						chess_piece = image 
					else:
						continue
				chessboard_matrix += chess_piece[-1]
				fen_notation += chess_piece[-1]
			column_index += tile_size

		if empty_tiles > 0:
			fen_notation += str(empty_tiles)
			empty_tiles = 0

		row_index += tile_size
		fen_notation += '/'
		chessboard_matrix += '/'

	return fen_notation[:-1], chessboard_matrix[:-1]



def get_xy(chessboard):
	indices = np.where(
		np.all(
			chessboard != (0, 0, 0), 
			axis = -1
			)
		)
	x_axis = indices[0][0]
	y_axis = indices[1][0]

	return x_axis, y_axis 



# -----------------------------------------------
# FUNCTIONS FOR CHECKING WHETHER THE CURRENT POSITION REPRESENTS A CHECK
# Whoever is reading this: I am sorry for the ugly code I've written down here
# but it works and I didn't have much time to come up with some more beautiful algorithm :)

def check(chessboard, king_color):

	if king_color == 'k':
		opposing_pieces = {
			'bishop' : 'B',
			'king' : 'K',
			'knight' : 'N',
			'queen' : 'Q',
			'rook' : 'R', 
			'pawn' : 'P'
		}
	elif king_color == 'K':
		opposing_pieces = {
			'bishop' : 'b',
			'king' : 'k',
			'knight' : 'n',
			'queen' : 'q',
			'rook' : 'r', 
			'pawn' : 'p'
			}

	king_x = 0
	king_y = 0

	for index, row in enumerate(chessboard):
		if king_color in row:
			king_y = row.index(king_color)
			break
		else:
			king_x += 1

	result = None
	lookup_functions = [leftright_diagonal_lookup, rightleft_diagonal_lookup, vertical_lookup, horizontal_lookup, knight_lookup]
	
	for i in range(5):
		result = lookup_functions[i](chessboard, king_color, king_x, king_y, opposing_pieces)
		if result == 'chess':
			return 'chess'

	return None


def vertical_lookup(chessboard, king_color, king_x, king_y, opposing_pieces):
	for i in range(king_x, 8):
		piece = chessboard[i][king_y]
		if piece == '*' or piece == king_color:
			continue
		elif piece not in opposing_pieces.values():
			break
		elif piece in opposing_pieces.values():
			if np.abs(king_x - i) == 1:
				if piece in [opposing_pieces['king'],opposing_pieces['queen'],opposing_pieces['rook']]:
					return 'chess'
				else:
					break
			else:
				if piece in [opposing_pieces['queen'], opposing_pieces['rook']]:
					return 'chess'
				else:
					break

	for i in reversed(range(king_x)):
		piece = chessboard[i][king_y]
		if piece == '*' or piece == king_color:
			continue
		elif piece not in opposing_pieces.values():
			break
		elif piece in opposing_pieces.values():
			if np.abs(king_x - i) == 1:
				if piece in [opposing_pieces['king'],opposing_pieces['queen'],opposing_pieces['rook']]:
					return 'chess'
				else:
					break
			else:
				if piece in [opposing_pieces['queen'], opposing_pieces['rook']]:
					return 'chess'
				else:
					break

	return None


def horizontal_lookup(chessboard, king_color, king_x, king_y, opposing_pieces):
	for i in range(king_y, 8):
		piece = chessboard[king_x][i]
		if piece == '*' or piece == king_color:
			continue
		elif piece not in opposing_pieces.values():
			break
		elif piece in opposing_pieces.values():
			if np.abs(king_y - i) == 1:
				if piece in [opposing_pieces['king'],opposing_pieces['queen'],opposing_pieces['rook']]:
					return 'chess'
				else:
					break
			else:
				if piece in [opposing_pieces['queen'], opposing_pieces['rook']]:
					return 'chess'
				else:
					break

	for i in reversed(range(king_y)):
		piece = chessboard[king_x][i]
		if piece == '*' or piece == king_color:
			continue
		elif piece not in opposing_pieces.values():
			break
		elif piece in opposing_pieces.values():
			if np.abs(king_y - i) == 1:
				if piece in [opposing_pieces['king'], opposing_pieces['queen'],opposing_pieces['rook']]:
					return 'chess'
				else:
					break 
			else:
				if piece in [opposing_pieces['queen'], opposing_pieces['rook']]:
					return 'chess'
				else:
					break

	return None


def knight_lookup(chessboard, king_color, king_x, king_y, opposing_pieces):
	possible_indices = zip(
			[
				king_x - 1, king_x - 1, king_x - 2, king_x - 2,
				king_x + 1, king_x + 1, king_x + 2, king_x + 2 
			],
			[
				king_y - 2, king_y + 2, king_y - 1, king_y + 1,
				king_y - 2, king_y + 2, king_y - 1, king_y + 1
			]
		)

	for idx_comb in possible_indices:
		try:
			if opposing_pieces['knight'] == chessboard[idx_comb[0]][idx_comb[1]]:
				return 'chess'
			else:
				continue
		except IndexError:
			continue

	return None


def leftright_diagonal_lookup(chessboard, king_color, king_x, king_y, opposing_pieces):
	x = king_x
	y = king_y
	for i in range(king_x, 8):
		if (x > 7 or y > 7) or (x < 0 or y < 0):
			break 
		piece = chessboard[x][y]
		if piece == '*' or piece == king_color:
			x += 1
			y += 1
			continue 
		elif piece not in opposing_pieces.values():
			break
		elif piece in opposing_pieces.values():
			if king_color == 'k':
				if (x - king_x == 1) and (y - king_y == 1):
					if piece in [opposing_pieces['pawn'], opposing_pieces['queen'], opposing_pieces['bishop'], opposing_pieces['king']]:
						return 'chess'
					else:
						break
				else:
					if piece in [opposing_pieces['queen'], opposing_pieces['bishop']]:
						return 'chess'
			elif king_color == 'K':
				if (king_x - 1 == 1) and (king_y - 1 == 1):
					if piece in [opposing_pieces['pawn'], opposing_pieces['queen'], opposing_pieces['bishop'], opposing_pieces['king']]:
						return 'chess'
					else:
						break
				else:
					if piece in [opposing_pieces['queen'], opposing_pieces['bishop']]:
						return 'chess'
	x = king_x 
	y = king_y

	for i in reversed(range(king_x)):
		if (x > 7 or y > 7) or (x < 0 or y < 0):
			break 
		piece = chessboard[x][y]
		if piece == '*' or piece == king_color:
			x -= 1
			y -= 1
			continue 
		elif piece not in opposing_pieces.values():
			break
		elif piece in opposing_pieces.values():
			if king_color == 'k':
				if (x - king_x == 1) and (y - king_y == 1):
					if piece in [opposing_pieces['pawn'], opposing_pieces['queen'], opposing_pieces['bishop'], opposing_pieces['king']]:
						return 'chess'
					else:
						break
				else:
					if piece in [opposing_pieces['queen'], opposing_pieces['bishop']]:
						return 'chess'
			elif king_color == 'K':
				if (king_x - x == 1) and (king_y - y == 1):
					if piece in [opposing_pieces['pawn'], opposing_pieces['queen'], opposing_pieces['bishop'], opposing_pieces['king']]:
						return 'chess'
					else:
						break
				else:
					if piece in [opposing_pieces['queen'], opposing_pieces['bishop']]:
						return 'chess'


	return None 


def rightleft_diagonal_lookup(chessboard, king_color, king_x, king_y, opposing_pieces):
	x = king_x
	y = king_y

	for i in range(king_x, 8):
		if (x > 7 or y > 7) or (x < 0 or y < 0):
			break 
		piece = chessboard[x][y]
		if piece == '*' or piece == king_color:
			x += 1
			y -= 1
			continue 
		elif piece not in opposing_pieces.values():
			break
		elif piece in opposing_pieces.values():
			if king_color == 'k':
				if (x - king_x == 1) and (y - king_y == -1):
					if piece in [opposing_pieces['pawn'], opposing_pieces['queen'], opposing_pieces['bishop'], opposing_pieces['king']]:
						return 'chess'
					else:
						break
				else:
					if piece in [opposing_pieces['queen'], opposing_pieces['bishop']]:
						return 'chess'
			elif king_color == 'K':
				if (king_x - x == 1) and (y - king_y == 1):
					if piece in [opposing_pieces['pawn'], opposing_pieces['queen'], opposing_pieces['bishop'], opposing_pieces['king']]:
						return 'chess'
					else:
						break
				else:
					if piece in [opposing_pieces['queen'], opposing_pieces['bishop']]:
						return 'chess'
	x = king_x 
	y = king_y

	for i in reversed(range(king_x)):
		if (x > 7 or y > 7) or (x < 0 or y < 0):
			break 
		piece = chessboard[x][y]
		if piece == '*' or piece == king_color:
			x -= 1
			y += 1
			continue 
		elif piece not in opposing_pieces.values():
			break
		elif piece in opposing_pieces.values():
			if king_color == 'k':
				if (x - king_x == 1) and (y - king_y == -1):
					if piece in [opposing_pieces['pawn'], opposing_pieces['queen'], opposing_pieces['bishop'], opposing_pieces['king']]:
						return 'chess'
					else:
						break
				else:
					if piece in [opposing_pieces['queen'], opposing_pieces['bishop']]:
						return 'chess'
			elif king_color == 'K':
				if (king_x - x == 1) and (y - king_y == 1):
					if piece in [opposing_pieces['pawn'], opposing_pieces['queen'], opposing_pieces['bishop'], opposing_pieces['king']]:
						return 'chess'
					else:
						break
				else:
					if piece in [opposing_pieces['queen'], opposing_pieces['bishop']]:
						return 'chess'

	return None 


def move_king(chessboard, king_color):
	# OVO NIJE DOBRO, AKO NE SREDIS DO SUTRA ONDA BRISI
	if king_color == 'k':
		opposing_pieces = {
			'bishop' : 'B',
			'king' : 'K',
			'knight' : 'N',
			'queen' : 'Q',
			'rook' : 'R', 
			'pawn' : 'P'
		}
	elif king_color == 'K':
		opposing_pieces = {
			'bishop' : 'b',
			'king' : 'k',
			'knight' : 'n',
			'queen' : 'q',
			'rook' : 'r', 
			'pawn' : 'p'
			}

	king_x = 0
	king_y = 0

	for index, row in enumerate(chessboard):
		if king_color in row:
			king_y = row.index(king_color)
			break
		else:
			king_x += 1

	new_chessboard = chessboard.copy()
	movements = [
		(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)
		]
	for movement in movements:
		x = king_x + movement[0]
		y = king_y + movement[1]
		if x < 0 or y < 0:
			continue
		try:
			tile = new_chessboard[x][y]
			if tile == '*' or tile in opposing_pieces.values():
				new_chessboard[x] = new_chessboard[x][:y] + king_color + new_chessboard[x][y + 1:]
				new_chessboard[king_x] = new_chessboard[king_x][:king_y] + '*' + new_chessboard[king_x][king_y + 1:]
				result = check(new_chessboard, king_color)
				if result:
					new_chessboard = chessboard.copy()
					continue
				elif not result:
					return 'not_check_mate'
			else:
				continue
		except IndexError:
			continue

	return 'check_mate'


# --------------------------
# RUNNING A PROGRAM
def run_program(path):
	# ----------------------------------------- 
	# Getting all information on chessboard image
	for file in os.listdir(path):
		if file.endswith('.png'):
			filename = os.path.join(path, file)
			chess_image = Image.open(filename)
	x_axis, y_axis = get_xy(np.array(chess_image))
	tile_size = get_tile_size(
		chessboard = np.array(chess_image), 
		first_x = x_axis, 
		first_y = y_axis
		)
	chessboard = extract_chessboard(
		chessboard = np.array(chess_image.convert('LA')), 
		first_x = x_axis, 
		first_y = y_axis, 
		tile_size = tile_size
		)
	# -------------------------------------------

	tiles, white_pieces, black_pieces = load_images(path)
	final_images = combine_images(tiles, white_pieces, black_pieces)
	final_images = rescale_images(tile_size, final_images)
	black_tile_color, white_tile_color = get_tile_colors(np.array(tiles['black']), np.array(tiles['white']))

	# -------------------------------------------
	fen, chessboard_matrix = get_fen(
		chessboard, 
		final_images,
		tile_size,
		black_tile_color,
		white_tile_color
		)
	# -------------------------------------------
	# Razmisli kako ovo da sredis :) 
	chessboard_matrix = chessboard_matrix.split('/')
	player = '-'
	try:
		if check(chessboard_matrix, 'k') == 'chess':
			player = 'W'
		else:
			if check(chessboard_matrix, 'K') == 'chess':
				player = 'B'
	except:
		player = None

	check_mate = None
	if player:
		if player == 'W':
			n_of_pieces = 0
			for row in chessboard_matrix:
				for ch in row:
					if ch.islower():
						n_of_pieces += 1
			try:
				check_mate = move_king(chessboard_matrix, 'k')
				if n_of_pieces > 1 and check_mate == 'check_mate':
					check_mate = None
			except:
				check_mate = None 
		elif player == 'B':
			n_of_pieces = 0
			for row in chessboard_matrix:
				for ch in row:
					if ch.isupper():
						n_of_pieces += 1
			try:
				check_mate = move_king(chessboard_matrix, 'K')
				if n_of_pieces > 1 and check_mate == 'check_mate':
					check_mate = None
			except:
				check_mate = None
			

	return x_axis, y_axis, fen, player, check_mate


if __name__ == "__main__":
	path = input()
	results = run_program(path)
	print(str(results[0]) + ',' + str(results[1]))
	print(results[2])
	if results[3]:
		print(results[3])
	else:
		print()
	if results[3] == '-':
		print(0)
	elif results[4] == 'not_check_mate':
		print(0)
	elif results[4] == 'check_mate':
		print(1)
	else:
		print()