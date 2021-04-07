import numpy as np 


def read_particles():
	'''
	Function for reading inputs.
	'''

	first_line = input().split()
	first_line = {
		'N' : int(first_line[0]),
		'S' : int(first_line[1]),
		'T' : int(first_line[2]),
		'P' : float(first_line[3])
		}

	particles = []
	for i in range(first_line['N']):
		particles.append(input())

	return first_line, particles


def clean_input(particles, n):
	'''
	Function creates two np.arrays from inputs. 
	One array consists of the current position of particles in 2-d coordinate system, while
	the other consists of their velocities in x and y dimensions. 

	Args:
		particles (list): list of lists of strings
		n (int): number of particles
	Returns:
		positions (np.array): NumPy array with positions of particles with shape (n, 2)
		movements (np.array): NumPy array with velocity vectors of particles with shape (n, 2)
	'''

	positions = np.zeros(shape = (n, 2))
	movements = np.zeros(shape = (n, 2))

	n = 0
	for line in particles:
		temp = [float(i) for i in line.split()]
		positions[n, :] = temp[:2]
		movements[n, :] = temp[-2:]
		n += 1

	return positions, movements


def beginning_of_time(positions, movements):
	'''
	The N number of particles were scattered around 0,0 by Gaussian distribution.
	The particles were then moved by Px and Py for each second. This function calculates
	how many seconds (K) ago was the beginning of time, given Px and Py for each particle.

	Since the standard normal distribution has [0,0] mean vector and [1, 1] vector variance,
	the solution is to calculate the variance of axes and revert particles to positions for which 
	the variance will be as closest to [1, 1] (expected variance) as possible.

	Args:
		positions (np.array): NumPy array with positions of particles with shape (N, Number of coordinates)
		movements (np.array): NumPy array with velocity vectors of particles with shape (N, Number of coordinates)
	Returns:
		seconds (int): The number of seconds during which particles moved from their starting position
	'''
	reversed_positions = np.copy(positions)
	seconds = 0 
	beginning = False 
	difference = np.var(reversed_positions, axis = 0)

	while beginning != True:
		seconds += 1
		reversed_positions = reversed_positions - movements 
		var = np.var(reversed_positions, axis = 0)
		if np.any(var - np.array([1, 1]) < difference) == True:
			difference = var - np.array([1, 1])
		else:
			beginning = True
			seconds -= 1
			return seconds


def bouncing(positions, movements, boundary, seconds, probability):

	# TODO ACCOUNT FOR POSSIBLY HITTING THE CORNER?


	
	'''
	Given the boundaries of a square (S), number of seconds (K), starting positions of 
	particles (Px, Py), and their movement vectors (Vx, Vy), the task is to find how many times 
	will particles bounce off the walls during K seconds. However, particles will not 
	loose any momentum or velocity after hitting the wall.

	Particles reflect off the wall such that the incoming angle is the same as its outgoing
	angle. However, it's not necessary to calculate angles since particles will only change the 
	sign of their Px if they hit the left/right wall and Py if they hit upper/bottom wall. 
	Furthermore, if particle hits the corner of the square perfectly, its Px and Py will 
	switch places and change their signs.

	Furthermore, given the probability one particle will NOT be absorbed by the wall after hitting it,
	the task is to find the expected number of particles after K seconds. Note that if one particle
	hits the wall multiple times, the expected probability of its survival is p^i where p is probability and 
	i is the number of bounces for that specific particle.

	Args:
		positions (np.array): NumPy array with positions of particles with shape (N of particles, 2)
		movements (np.array): NumPy array with velocity vectors of particles (N of particles, 2)
		boundary (float): boundary of square
		seconds (int): Number of seconds during which the particles will move
	Returns:
		bounces (int): The total number of bounces during K seconds
		survived (float): The expected number of particles after K seconds
	'''

	negative_boundary = np.negative(boundary)
	positive_boundary = boundary
	new_positions = positions.copy()
	bounce_movements = movements.copy()
	bounces = 0
	probabilities = np.array(
		np.ones(
			shape = (np.shape(new_positions)[0], 1)
			)
		) # Starting probability of survival is 1 for each particle


	for i in range(1, seconds + 1):
		new_positions = new_positions + bounce_movements
		while np.any(np.abs(new_positions) > positive_boundary) == True: 
			positive_x_bounces = np.where(new_positions[:, 0] > positive_boundary)[0]
			negative_x_bounces = np.where(new_positions[:, 0] < negative_boundary)[0]
			positive_y_bounces = np.where(new_positions[:, 1] > positive_boundary)[0]
			negative_y_bounces = np.where(new_positions[:, 1] < negative_boundary)[0]


			bounces = bounces + positive_x_bounces.size + negative_x_bounces.size + positive_y_bounces.size + \
			negative_y_bounces.size

			positive_x_difference = new_positions[positive_x_bounces, 0] - positive_boundary
			negative_x_difference = new_positions[negative_x_bounces, 0] - negative_boundary
			positive_y_difference = new_positions[positive_y_bounces, 1] - positive_boundary
			negative_y_difference = new_positions[negative_y_bounces, 1] - negative_boundary

			new_positions[positive_x_bounces, 0] = positive_boundary - positive_x_difference 
			new_positions[negative_x_bounces, 0] = negative_boundary - negative_x_difference
			new_positions[positive_y_bounces, 1] = positive_boundary - positive_y_difference 
			new_positions[negative_y_bounces, 1] = negative_boundary - negative_y_difference 

			probabilities[positive_x_bounces] = probabilities[positive_x_bounces] * probability
			probabilities[negative_x_bounces] = probabilities[negative_x_bounces] * probability
			probabilities[positive_y_bounces] = probabilities[positive_y_bounces] * probability
			probabilities[negative_y_bounces] = probabilities[negative_y_bounces] * probability

			# Change the sign of movement vectors
			bounce_movements[positive_x_bounces, 0] = np.negative(bounce_movements[positive_x_bounces, 0])
			bounce_movements[negative_x_bounces, 0] = np.negative(bounce_movements[negative_x_bounces, 0])
			bounce_movements[positive_y_bounces, 1] = np.negative(bounce_movements[positive_y_bounces, 1])
			bounce_movements[negative_y_bounces, 1] = np.negative(bounce_movements[negative_y_bounces, 1])

	expected_survived = np.sum(probabilities)

	return bounces, expected_survived



def run_program():
	first_line, particles = read_particles()
	positions, movements = clean_input(particles, first_line['N'])
	seconds = beginning_of_time(positions, movements)
	bounces, survived = bouncing(
		positions, 
		movements, 
		first_line['S'], 
		first_line['T'], 
		first_line['P']
		)

	return seconds, bounces, survived

if __name__ == '__main__':
	seconds, bounces, survived = run_program()
	print(seconds, bounces, survived)