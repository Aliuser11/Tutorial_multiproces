"""Basics of multiprocessing"""

#import multiprocessing
#import time

#def task():
#    print(f" Sleeping for 0,5 sec.")
#    time.sleep(0.5)
#    print(f" Finishing sleeping")

#if __name__ == '__main__':
#    start_time = time.perf_counter()
#    process1 = multiprocessing.Process(target = task)
#    process2 = multiprocessing.Process(target = task)

#    process1.start() # to star the process cause it does not run immediately
#    process2.start()

#    process1.join()
#    process2.join()

#    finish_time = time.perf_counter()
    #print(f" Program finished in {finish_time-start_time} seconds.")

""" with a for loop"""

#import time
#import multiprocessing

#def task():
#    print(f" Sleeping for 0,5 sec.")
#    time.sleep(0.5)
#    print(f" Finishing sleeping")

#if __name__ == "__main__":
#    start_time = time.perf_counter()
#    processes = []

#    for i in range(10):
#        process = multiprocessing.Process(target = task)
#        process.start()
#        processes.append(process)

#    for process in processes:
#        process.join()

#    finish_time = time.perf_counter()
#    print(f"Program finishef in {finish_time-start_time} sec.")
#    print()

import logging
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
from multiprocessing import Process
from queue import Queue 

def duble(number):
    result = number * 2
    proc = os.getpid()
    print("{0} doubled to {1} by process id: {2}".format(number, result, proc))

if __name__ == "__main__":
    numbers = [5, 10, 15, 20 ,25]
    procs =[]

    for index, number, in enumerate(numbers):
        proc = Process(target = duble, args = (number,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

import os 
from multiprocessing import Process, current_process

def double(number):
    result = number * 3
    proc_name = current_process().name
    print('{0} double to {1} by: {2}'. format( number, result, proc_name))

if __name__ == "__main__":
    numbers = [5, 10, 15, 20 ,25]
    procs = []

    #procs = Process(target = double, args = (5,))
     
    for index, number, in enumerate(numbers):
        proc = Process(target = double, args = (number,))
        procs.append(proc)
        proc.start()

    proc = Process(target = double, name = 'test', args = (2,))
    proc.start()
    procs.append(proc)

    for proc in procs:
        proc.join()

"""pool class"""

#from multiprocessing import Pool, TimeoutError
#import time
#import os

#def f(x):
#    return x * x

#if __name__ == "__main__":
#    with Pool(processes = 4) as pool:
#        print(pool.map(f, range(10)))

#        for i in pool.imap_unordered(f, range(10)):
#            print(i)

#        result = pool.apply_async(f, (20,))
#        print(result.get(timeout = 1))

#        result = pool.apply_async(os.getpid, ())
#        print(result.get(timeout = 1))

#        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
#        print()
#        print([result.get(timeout = 1) for result in multiple_results])

#        result = pool.apply_async(time.sleep, (10,))

#        try:
#            print(result.get(timeout = 1))
#        except TimeoutError:
#            print("lack of patience")
#        print('pool available for more work')
#    print('pool no longer available')



#import multiprocessing
#from multiprocessing import Pool 
#import time

#def double(number):
    #return number * 3

#if __name__ == "__main__":
#   with Pool(processes = 4) as pool:
#        result = pool.apply_async(double, (10,))
#        print(result.get(timeout = 1))
#        print(pool.map(double, range(10)))

#        it = pool.imap(double, range(10))
#        print(next(it))
#        print(next(it))
#        print(it.next(timeout = 1))
#        result = pool.apply_async(time.sleep, (10,))
#        print(result.get(timeout = 1))

#if __name__ == "__main__":
#    with Pool(processes = 3) as pool:
#        result = pool.apply_async(double, (25,))
#        print(result)
#        print(result.get(timeout = 1))


#def cube(x):
#    return x ** 3

#if __name__ == "__main__":
#    pool = multiprocessing.Pool(3) #number of process that can be run in one time
#    start_time = time.perf_counter()

"""the apply_async() function does not return the result but an object that we can use, get(), to wait for the task to finish and retrieve the result"""
    #processes = [pool.apply_async(cube, args = (x, )) for x in range(1, 10)]
    #result = [p.get() for p in processes]
    #finish_time = time.perf_counter()

    #print(f"Finishing program time : {finish_time-start_time} seconds") # 0.3913166000000001 seconds
    #print(result)
    #print()

#""" the same but with the pool.map() function. that hides START and JOIN  itself"""
#if __name__ == "__main__":
#    pool = multiprocessing.Pool(3)
#    start_time = time.perf_counter()
#    result = pool.map(cube, range (1, 10))
#    finish_time = time.perf_counter()

#    print(f" Program finished in {finish_time-start_time} seconds") #0.3767782999999998 seconds
#    print(result)
#    print()

#"""the same multiprocessing but with concurrent.futures"""
#import concurrent.futures

#def cube(x):
#    return x ** 5

#if __name__ == "__main__":
#    with concurrent.futures.ProcessPoolExecutor(3) as executor:
#        start_time = time.perf_counter()
#        result = list(executor.map(cube, range(1, 10)))
#        finish_time = time.perf_counter()
#    print(f"Finishing in {finish_time - start_time}") #0.5235086999999998
#    print(f"results : {result}")
#    print()

"""multithreading with the ThreadPoolExecutor"""

#if __name__ == "__main__":
#    with concurrent.futures.ThreadPoolExecutor(3) as executor:
#        start_time = time.perf_counter()
#        result = list(executor.map(cube, range(1, 10)))
#        finish_time = time.perf_counter()
#    print(f"Finishing in {finish_time - start_time}") #0.01547289999999979  <--> NAJSZYBSZE
#    print(f"results : {result}")
#    print()


"""yet another tutorial"""

#from multiprocessing import Pool
#def function(x):
#    return x * x 
#if __name__ == "__main__":
#    with Pool(5) as p:
#        print(p.map(function, [1, 2, 3]))


#example of multiprocess with START() and JOIN()
#from multiprocessing import Process
#def fun(name):
#    print(f"Hello, {name}")
#if __name__ =="__main__":
#    p = Process(target = fun, args = ('Alice', ))
#    p.start()
#    p.join()

"""individual process is shown below"""
#import os

#def info(title):
#    print(f"{title}")
#    print('module name; ', __name__)
#    print('parent process:', os.getppid())
#    print('parent id:', os.getpid())

#def fun(name):
#    info('function fun')
#    print('hello', name)

#if __name__ == "__main__":
#    info('main line')
#    p = Process(target = fun, args = ('Alice',))
#    p.start()
#    p.join()
#    print()

"""the set_start_method must be used not more that once in the program"""
"""SPAWN child process will only inherit those resources necessary to run the process objects run() method"""

#import multiprocessing as mp
#def foo(q):
#    q.put('Hello')

#if __name__ == "__main__":
#    #mp.set_start_method('spawn')
#    q = mp.Queue()
#    p = mp.Process(target = foo, args = (q,))
#    p.start()
#    print(q.get())
#    p.join()
#    print()

#if __name__ == "__main__":
#    ctx = mp.get_context('spawn')
#    q = ctx.Queue()
#    p = ctx.Process(target = foo, args = (q,))
#    p.start()
#    print(q.get())
#    p.join()
#print()

""" multiprocessing supports two types of communication channel between processes: queues and pipes
For passing messages one can use Pipe() (for a connection between two processes) or a queue (which allows multiple producers and consumers)."""

"""queue"""
#from multiprocessing import Process, Queue

#def function(q):
#    q.put([42 , None, 'Hello'])

#if __name__ == "__main__":
#    print('queue')
#    q = Queue()
#    p = Process(target = function, args = (q,))
#    p.start()
#    print(q.get())
#    p.join()
#    print()

"""pipes"""
#from multiprocessing import Process, Pipe
#def f(conn):
#    conn.send([66, None, "Hey"])
#    conn.close()

#if __name__ == "__main__":
#    parent_conn, child_conn = Pipe()
#    p = Process(target = f, args = (child_conn,))
#    p.start()
#    print(parent_conn.recv())
#    p.join()
    
#from multiprocessing import Process, freeze_support
#from multiprocessing import Pipe

#def f():
#    print("hey")

#if __name__ == "__main__":
#    freeze_support() # without this line --> it will return RuntimeError
#    Process(target = f). start() 


"""synchronization"""
"""Locks"""

#from multiprocessing import Process, Lock

#def printer(item, lock):

#    lock.acquire()
#    try:
#        print(item)
#    finally:
#        lock.release()

#if __name__ == "__main__":
#    lock = Lock()
#    items = ['jablko', 'picie', 10]
#    for item in items:
#        p = Process(target = printer, args = (item, lock))
#        p.start()

#from multiprocessing import Process, Lock

#def fun(l, i):
#    l.acquire()
#    try:
#        print('hello')
#    finally:
#        l.release()

#if __name__ == "__main__":
#    lock = Lock()
#    for number in range(10):
#        Process(target = fun, args = (lock, number)).start()


"""logging"""
#print('logging')
#import logging
#import multiprocessing
#from multiprocessing import Process, Lock

#def printer(item, lock):

#    lock.acquire()
#    try:
#        print(item)
#    finally:
#        lock.release()

#if __name__ == "__main__":
#    lock = Lock()
#    items = ['jablko', 'picie', 10]
#    multiprocessing.log_to_stderr()
#    logger = multiprocessing.get_logger()
#    logger.setLevel(logging.INFO) 

#    for item in items:
#        p = Process(target = printer, args = (item, lock))
#        p.start()


'''process communication'''

#from multiprocessing import Process, Queue

#sentiel = -1
#def creator(data, q):
#    print('creating and putting into queue')
#    for item in data:
#        q.put(item)

#def my_consumer(q):
#    while True:
#        data = q.get()
#        print('data found to be processes  {}'. format(data))
#        processed = data * 2
#        print(processed)

#        if data is sentiel:
#            break

#if __name__ == "__main__":
#    q = Queue()
#    data = [5, 10 ,13, -2]
#    process_one = Process(target = creator, args = (data, q))
#    process_two = Process(target = my_consumer, args = (q,))

#    process_one.start()
#    process_two.start()

#    q.close()
#    q.join_thread()

#    process_one.join()
#    process_two.join()

#from multiprocessing import Process, Lock 
#from multiprocessing.sharedctypes import Value, Array
#from ctypes import Structure, c_double 

#print()
#class Point(Structure):
#    _fields_ = [('x', c_double), ('y', c_double)]

#def modify(n, x, s, A):
#    n.value **= 2
#    x.value **= 2
#    s.value = s.value.upper()
#    for a in A:
#        a.x **= 2
#        a.y **= 2

#if __name__ == '__main__':
#    lock = Lock()

#    n = Value('i', 7)
#    x = Value(c_double, 1.0/3.0, lock = False)
#    s = Array('c', b'hello world', lock = lock)
#    A = Array(Point, [(1.875,-6.25), (-5.75,2.0), (2.375,9.5)], lock = lock)

#    p = Process(target = modify, args = (n, x, s, A))
#    p.start()
#    p.join()

#    print(n.value)
#    print(x.value)
#    print(s.value)
#    print([(a.x, a.y) for a in A])

#from multiprocessing import Pipe
#a, b = Pipe()
#a.send([1, 'hey', None])
#b.recv()
#print(b.send_bytes(b'thanks'))
#print(a.recv_bytes())

#import array
#arr1 = array.array('i', range(5))
#arr2 = array.array('i', [0]*10)
#a.send_bytes(arr1)
#count = b.recv_bytes_into(arr2)
#assert count == len(arr1) * arr1.itemsize
#print(arr2)
#print(arr1)

"""shared memory"""

#from multiprocessing import Process, Value, Array

#def f(n, a):
#    n.value = 3.1415927
#    for i in range(len(a)):
#        a[i] = -a[i]

#if __name__ == '__main__':
#    num = Value('d', 0.0)
#    arr = Array('i', range(10))

#    p = Process(target=f, args=(num, arr))
#    p.start()
#    p.join()

#    print(num.value)
#    print(arr[:])

"""Server process"""

#from multiprocessing import Process, Manager  # slower than shared memory byt more flexible
#def fun(d, l):
#    d[1] = '1'
#    d[2] = 2
#    d[3] = None
#    l.reverse()

#if __name__ == "__main__":
#    with Manager() as m:
#        d = m.dict()
#        l = m.list(range(10))

#        p = Process(target = fun, args = (d ,l)) 
#        p.start()
#        p. join()
#        print('d', d)
#        print('l',l)

"""Global| BaseManager"""

#import multiprocessing
#from multiprocessing.managers import BaseManager
#manager = BaseManager (address = ('', 500), authkey = b'abc')
#server = manager.get_server()
#server.serve_forever()

#manager = BaseManager(address = ('127.0.0.1', 500), autkey = b'abc')
#manager. connect()

#manager = multiprocessing.Manager()
#Global = manager.Namespace()
#Global.x = 10
#Global.y = 'hey'
#Global._z = 12.3
#print(Global)

'''Customized managers| BaseManager'''

#from multiprocessing.managers import BaseManager
#class Math:
#    def add(self, x, y):
#        return x + y 
#    def mul(self, x, y):
#        return x * y

#class MyManager(BaseManager) :
#    pass
#MyManager.register('Maths', Math)

#if __name__ == "__main__":
#    with MyManager() as manager:
#        maths = manager.Maths()
#        print(maths.add(4 ,3))
#        print(maths.mul(7, 8))

'''remote manager |Running the following commands creates a server for a single shared queue which remote clients can access'''

#from multiprocessing.managers import BaseManager
#from queue import Queue
#queue = Queue()
#class QueueManager(BaseManager): pass 
#QueueManager.register('get_queue', callable = lambda: queue) 
#m = QueueManager(address = ('', 500), authkey = b'abra')
#s = m.get_server()
#s.serve_forever()

"""client can access the server as follow"""

#from multiprocessing.managers import BaseManager
#class QueueManager(BaseManager): pass
#QueueManager.register('get_queue')
#m = QueueManager(address = ('foo.bar.org', 500), authkey = b'abra')
#m.connect()
#queue = m.get_queue()
#queue.put('hey')

"""Local processes can also access that queue, using the code from above on the client"""

#from multiprocessing import Process, Queue
#from multiprocessing.managers import BaseManager

#class Worker(Process):
#    def __init__(self, q):
#        self.q = q
#        super().__init__()
#    def run(self):
#        self.q.put('local hello')
#queue = Queue()
#w = Worker(queue)
#w.start()

#class QueueManager(BaseManager): pass 
#QueueManager.register('get_queue', callable = lambda : queue)
#m = QueueManager(address = ('', 500), authkey = b'abra')
#s = m.get_server()
#s.serve_forever()

"""proxy objects"""

#from multiprocessing import Manager

#manager = Manager()
#l = manager.list([i*i for i in range(10)])
#print(l)
#print(repr(l))


"""Reference| run()"""

#from multiprocessing import Process 
#import multiprocessing, time, signal

#p = Process(target = print, args = [1])
#p.run()
#p = Process(target = print, args = (1,))
#p.run()

#p = multiprocessing.Process(target = time.sleep, args = (100,))
#print(p, p.is_alive())
#p.start()
#print(p, p.is_alive())
#p.terminate()
#time.sleep(0.1)
#print(p, p.is_alive())
#p.exitcode == -signal.SIGTERM


"""using joblib | """

#from joblib import Parallel, delayed
#import time

#def cube(x):
#    return x**2
#start_time = time.perf_counter()
###result = Parallel(n_jobs = 3)(delayed(cube)(i) for i in range(1, 100))
###or like this:
#result = Parallel(n_jobs = 3)((cube, (i,) ,{})for i in range(1, 100))
###result = Parallel(n_jobs = 3, prefer = 'threads')(delayed(cube)(i) for i in range(1, 100))
#finish_time = time.perf_counter()
#print(f"Finishing in {finish_time - start_time}")
#print(f"results : {result}")
#print()


"""QUEUE"""

#import time
#from multiprocessing import Process, Queue, current_process, freeze_support

#def worker(input, output):
#    for func, args in iter(input.get, 'STOP'):
#        result = calculate(func, args)
#        output.put(result)

#def calculate(func, args):
#    result = func(*args)
#    return '%s says that %s%s = %s' % \
#        (current_process().name, func.__name__, args, result)

#def mul(a, b):
#    time.sleep(0.5)
#    return a * b

#def plus(a, b):
#    time.sleep(0.5)
#    return a + b

#def test():
#    NUMBER_OF_PROCESSES = 4
#    TASKS1 = [(mul, (i, 6)) for i in range(5)]
#    TASKS2 = [(plus, (i, 9)) for i in range(4)]

#    task_queue = Queue()
#    done_queue = Queue()
#    for task in TASKS1:
#        task_queue.put(task)
#    for task in TASKS2:
#        task_queue.put(task)

#    for i in range(NUMBER_OF_PROCESSES):
#        Process(target = worker, args = (task_queue, done_queue)).start()

#    print('Unordered results:')

#    for i in range(len(TASKS1)):
#        print('\t', done_queue.get())
#    for i in range(len(TASKS2)):
#        print('\t', done_queue.get())

#    for i in range(NUMBER_OF_PROCESSES):
#        task_queue.put('STOP')

#if __name__ == '__main__':
#    freeze_support()
#    test()


'''pool'''

import multiprocessing
import time

def calculate(func, args):
    result = func(*args)
    return '%s says that %s%s = %s' % (
        multiprocessing.current_process().name,
        func.__name__, args, result)

def calculatestar(args):
    return calculate(*args)

def mul(a, b):
    time.sleep(0.5)
    return a * b

def plus(a, b):
    time.sleep(0.5)
    return a + b

def test():
    PROCESSES = 4
    print('Creating pool with %d processes\n' % PROCESSES)

    with multiprocessing.Pool(PROCESSES) as pool:
        TASKS = [(mul, (i, 4)) for i in range(6)] + \
                [(plus, (i, 5)) for i in range(5)]

        results = [pool.apply_async(calculate, t) for t in TASKS]
        imap_it = pool.imap(calculatestar, TASKS)
        imap_unordered_it = pool.imap_unordered(calculatestar, TASKS)

        print('Ordered results using pool.apply_async():')
        for r in results:
            print('\t', r.get())
        print()

        print('Ordered results using pool.imap():')
        for x in imap_it:
            print('\t', x)
        print()

        print('Unordered results using pool.imap_unordered():')
        for x in imap_unordered_it:
            print('\t', x)
        print()

        print('Ordered results using pool.map() --- will block till complete:')
        for x in pool.map(calculatestar, TASKS):
            print('\t', x)
        print()
