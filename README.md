# mountain-car

Reinforcement learning applies Q-learning

This library contains an executable python file: [q_learning.py](https://github.com/kmair/mountain-car/blob/master/python/q_learning.py) 
within the python folder.  
It implements Q-learning algorithm to train the modeled car to reach the mountain top based on its position on it.  
Please do not replicate if you are enrolled in this course.


## Usage

python q_learning.py <mode> <weight_out_location> <returns_out_location> <episodes> <max_iterations> <epsilon> <gamma> <learning_rate>

Example:
python q_learning.py raw 'Output files/weight.out' 'Output files/returns.out' 4 200 0.05 0.99 0.01

## Convergence

Check the [Empirical solutions](Empirical solutions.xlsx) file for convergence. 
For instance, one set of parameters is:
- mode: tile
- episodes: 40
- max_iterations: 2000