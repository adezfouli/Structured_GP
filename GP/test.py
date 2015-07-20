import Queue
import threading
import urllib2

# called by each thread
import time
import numba
import numpy as np

class A:
    pass

@numba.jit("int32(double[:], int32)")
def get_url(q, url, m, a):
    print q[q]
    print a.b
    time.sleep(3)

theurls = np.array([0,1])

# q = Queue.Queue()

for u in theurls:
    a = A()
    a.b = 3
    t = threading.Thread(target=get_url, args = (theurls, u, [1,2,3], a))
    t.start()

