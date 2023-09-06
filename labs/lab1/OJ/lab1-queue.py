#%%

class Queue(object):
    def __init__(self):
        self._elements = []
        self._size = 0

    def is_empty(self):
        if self._size <= 0:
            return True
        else:
            return False

    def insert(self,x):
        self._size += 1
        self._elements.append(x)

    def pop(self):
        if self.is_empty():
            #raise Exception("Queue underflow!")
            return "Queue underflow!"
        self._size -= 1
        self._elements.pop(0)

    def get_size(self):
        return self._size


#%%

stk = Queue()
stk.insert('a')
stk.insert('b')
stk.insert('c')
print(stk.get_size())
print(stk.is_empty())
print(stk.pop())
print(stk.get_size())
print(stk.pop())
print(stk.pop())
print(stk.pop())

print(stk.get_size())
