import copy
def matrix_add(c,e):
    a=copy.deepcopy(c)
    b=copy.deepcopy(e)
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j]+=b[i][j]
    return a

def matrix_mul(c,e):
    a
    