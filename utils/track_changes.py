import numpy as np
from queue import Queue

class track_changes:
    def __init__(self, iterations):
        self.q = Queue(maxsize=iterations)
        self.param = []
        
        
    def compare(self, item, param):
        self.item = item

        if self.q.full():
            exit()

        self.param.append(param)
        if self.q.qsize() >= 1:
            if len(set(list(self.q.queue)[-1]) ^ set(item)) != 0:
                #print("**", set(list(self.q.queue)[-1]) ^ set(item))
                self.q.get()
        self.q.put_nowait(item)
        #print(list(self.q.queue))

        if self.q.qsize() == self.q.maxsize:
            return self.param[-self.q.maxsize]

        
            
        #return self

if __name__ == '__main__':
    obj = track_changes(3)
    obj.compare([1,2,3], 0.1)
    obj.compare([1,2,3], 0.2)
    obj.compare([1,2,3], 0.3)
    obj.compare([1,2,3], 0.4)
    print(obj.compare([1,2,3], 0.5))
