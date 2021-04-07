# BigBang

This is the explanation of the *BigBang Task*. You can find me solution [here](https://github.com/emirdemic/Petnica-Summer-Institute-Machine-Learning---Homework-Assignmnets/blob/main/big-bang/big_bang.py) which got 960/1000 points. Unfortunately, I knew immediately what I got wrong and
what could have been improved, but I didn't have any time remaining.

## Description:

At the **beginning of time** (***K*** seconds ago), ***N*** particles were scattered
around the origin point by normal distribution around every axis.
You're given positions of these particles after they were moving for ***K*** (whole) seconds.
* Particles are in a two-dimensional space
* Each particle has position and velocity described by two coordinates - ***Px, Py*** and  ***Vx, Vy*** respectively.
* ***Vx*** and ***Vy*** represent the velocity vector and describe how many units per second the particle moves in the ***x*** and ***y*** direction
* Particles move one velocity vector per second.
* Particles do not lose momentum or velocity and they do not collide with each other.

Given these particles, you need to answer the following questions:


A) How many seconds ago (***K***, integer) was the **beginning of time**, given the particles at input as the **final** position?
Assume that all particles were moving in a straight line along their velocity vectors since the beginning of time and nothing had changed their direction.


B) All particles exist in an ***2S*** x ***2S*** unit square which has center at origin (0,0). In other words particles are bound by [-S, S] around each axis. Particles cannot exit the square, but when they hit the square they reflect off its inner sides without any momentum loss.
Given the particles at input as the **starting** position, how many times have all particles hit the sides of the square in the following ***T*** seconds?

Particles reflect off the wall such that the incoming angle of the particle with the wall is the same as its outgoing angle:
![Image of reflection or text describing it](https://petljamediastorage.blob.core.windows.net/psiml/Images/bigbang-image1.png)


C) Each time a particle reflects off the square, it has a probability ***P*** of *NOT* being absorbed by the square. What is the expected number of remaining particles after ***T*** seconds?

## Input format:
All inputs will be given through the standard input. The first line of input consists of 4 numbers: ***N S T P***.

***N*** (integer) represents the number of particles. ***S*** (integer) is the half-length of square sides, while center of the square is at origin (0,0). ***T*** (integer) stands for the time in seconds during which particles reflect off the square. Finally, ***P*** (float) is the probability of the particle reflecting and not being absorbed by the square.

The following ***N*** lines of the input represent the particles in format: ***Px Py Vx Vy***. Where ***Px*** (float) and ***Py*** (float) stand for position of the particle. ***Vx*** (float) and ***Vy*** (float) represent the velocity vector of the particle.

## Output format:
Output answers to subtasks as 3 numbers ***Ta***, ***Tb***, ***Tc***. The expected output format is to have 3 numbers in one line of output.
If you calculate the solution to only one or two subtasks, return -1 as a result for other subtasks.
For example, if you calculate only the result of subtask 2 you would return `-1 Tb -1`.
- ***Ta*** (integer) is answer to the **A)** subtask - how many seconds ago was the beginning of time.
- ***Tb*** (integer) is the answer to the **B)** subtask - how many times have particles hit the square sides.
- ***Tc*** (float) is the answer to the **C)** subtask - what is the expected number of remaining particles.


## Example:
Input:
```
2 10 2 0.5
9.1 2.9 3.0 1.0
-6.1 -6.2 -2.0 -2.0
```

Output:
```
3 3 0.75
```

Explanation:

- The particles are closest to the center *{P1: (0.1, -0.1), P2: (-0.1, -0.2)}* 3 seconds
before their initial position.
- 2 Seconds after the initial position the first particle
hit the wall once (on the x axis in the first second) and the second particle hit
the wall twice (on x axis and the y axis in the second second). Thus, the total number of hits is 3.
-  Considering that the first particle has hit the wall once and the second particle has hit the world twice, the expected number of remaining particles is 0.75.


## Scoring

- Correct result for task 1 brings 40 points per test case.
- Correct result for task 2 brings 30 points per test case.
- Correct result for task 3 brings 30 points per test case.
- Probabilities in task 3 are considered correct if they don't differ from the expected values by more than 0.05


## Datasets
There are two data sets:
* *Public data set* is used for developing your solution.
After you submit your solution, you will be able to see how well your solution
performs against this data set. *Public data set* is not used for calculating
the final score. Public data set is available [***here***](https://petljamediastorage.blob.core.windows.net/psiml/2020/BigBang/public-dataset.zip).
* *Private data set* is used for testing your solution.
The final score will be measured against this data set.
*Private data set* and the final score will be available after the homework finishes.
*Private data set* contains different data than the *public data set*,
but the type of data is roughly the same (the same constraints apply).
