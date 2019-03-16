class A:
    def __init__(self, x):
        self.x = x

class B(A):
    def __init__(self, x, y):
        super().__init__(x)
        self.y = y

class C(B):
    def __init__(self, x, y, z):
        super().__init__(x ,y)
        self.z = z

inst = C(1,2,3)
print(inst.x, inst.y, inst.z)
