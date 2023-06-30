"""Basic multiprocessing"""
import multiprocessing
import time

def task():
    print(f" Sleeping for 0,5 sec.")
    time.sleep(0.5)
    print(f" Finishing sleeping")

if __name__ == '__main__':

    process1 = multiprocessing.Process(target = task)
    process2 = multiprocessing.Process(target = task)

    process1.start() # to star the process cause it does not run immediately
    process2.start()

    process1.join()
    process2.join()

    finish_time = time.perf_counter()
    print(f" Program finished in {finish_time-finish_time} seconds.")


