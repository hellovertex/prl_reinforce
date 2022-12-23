import time

import ray


@ray.remote
def no_work(x):
    return x


start = time.time()
num_calls = 1000
[ray.get(no_work.remote(x)) for x in range(num_calls)]
print("per task overhead (ms) =", (time.time() - start) * 1000 / num_calls)