import time
import multiprocessing
# with a for loop
def task():
    print(f" Sleeping for 0,5 sec.")
    time.sleep(0.5)
    print(f" Finishing sleeping")

if __name__ == "__main__":
    start_time = time.perf_counter()
    processes = []

    for i in range(10):
        process = multiprocessing.Process(target = task)
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    finish_time = time.perf_counter()
    print(f"Program finishef in {finish_time-start_time} sec.")
    print()


def cube(x):
    return x**3

if __name__ == "__main__":
    pool = multiprocessing.Pool(3) #number of process that can be run in one time
    start_time = time.perf_counter()
    """the apply_async() function does not return the result but an object that we can use, get(), to wait for the task to finish and retrieve the result"""
    processes = [pool.apply_async(cube, args = (x,)) for x in range(1, 100)]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()

    print(f"Finishing program time : {finish_time-start_time} seconds")
    print(result)
    print()

""" the same but with the pool.map() function. that hides START and JOIN  itself"""

if __name__ == "__main__":
    pool = multiprocessing.Pool(3)
    start_time = time.perf_counter()
    result = pool.map(cube, range (1, 100))
    finish_time = time.perf_counter()

    print(f" Program finished in {finish_time-start_time} seconds")
    print(result)
    print()

"""the same multiprocessing but with concurrent.futures"""
import concurrent.futures

def cube(x):
    return x**5

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(3) as executor:
        start_time = time.perf_counter()
        result = list(executor.map(cube, range(1, 100)))
        finish_time = time.perf_counter()
    print(f"Finishing in {finish_time - start_time}")
    print(f"results : {result}")
    print()

"""multithreading with the ThreadPoolExecutor"""
if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(3) as executor:
        start_time = time.perf_counter()
        result = list(executor.map(cube, range(1, 100)))
        finish_time = time.perf_counter()
    print(f"Finishing in {finish_time - start_time}")
    print(f"results : {result}")
    print()