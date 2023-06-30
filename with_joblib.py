
"""using joblib | """
from joblib import Parallel, delayed
import time

def cube(x):
    return x**2
start_time = time.perf_counter()
##result = Parallel(n_jobs = 3)(delayed(cube)(i) for i in range(1, 100))
##or like this:
result = Parallel(n_jobs = 3)((cube, (i,) ,{})for i in range(1, 100))
##result = Parallel(n_jobs = 3, prefer = 'threads')(delayed(cube)(i) for i in range(1, 100))
finish_time = time.perf_counter()
print(f"Finishing in {finish_time - start_time}")
print(f"results : {result}")
print()
