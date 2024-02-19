import tensorflow as tf
import numpy as np

import time

data = tf.abs(np.array([1, 2, 3]))

start = time.perf_counter_ns()

data = tf.abs(np.array([1, 2, 3]))

end = time.perf_counter_ns()

elapsed = end - start
print(elapsed)
