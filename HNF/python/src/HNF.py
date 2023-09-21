from MyMatrix import MyMatrix
from Utility import *
from sympy import nextprime
from math import sqrt


def hnf_mod_D(A:MyMatrix, d=None) -> MyMatrix:
    """
    Computes the HNF of a matrix A with the Modulo Determinant Algorithm. 

    @param A (MyMatrix): a nonsingular integer matrix.  
    @return (MyMatrix): the HNF of A.
    """
    B = A.copy()
    m = B.getRows()
    D = [None]
    if d is None:
        for _ in range(m):
            D.append(abs(A.getDet()))
    else:
        for _ in range(m):
            D.append(abs(d))
    D.append(None)
    for i in range(m):
        D_i = MyMatrix.get_e_i(m, i, new_value=D[1])
        B.addColumn(D_i.getList())

    n = B.getCols()
    for i in range(1, m+1):
        B.generateZeroRowMod(i, D)
        for j in range(1, i):
            B.reduceColumnMod(j, i, d=D)
        D[i+1] = D[i] // B[i, i]

    for i in range(n//2):
        B.removeColumn(n-i)

    return B

def hnf_basic(A: MyMatrix) -> MyMatrix:
     """
    Computes the HNF of a matrix A with the most basic algorithm. 

    @param A (MyMatrix): a nonsingular integer matrix.  
    @return (MyMatrix): the HNF of A.
    """
    B = A.copy()
    n = B.getRows()
    D = abs(A.getDet())
    for row in range(1, n + 1):
        if B[row, row] == 0:
            B[row, row] = D
        elif B[row, row] < 0:
            B.invertColumn(row)# mutliply with -1
        for col in range(row + 1, n + 1):
            B.generateZero(row, col)
        if B[row, row] == 0:
            B[row, row] = D
        D = D // B[row, row]

    for c in range(1, n+1):
        B.reduceColumn(c)
    return B


def addRow(B: MyMatrix, H_B: MyMatrix, a_t: MyMatrix) -> MyMatrix:
    """
    Computes a vector x such that [H_B^T|x^T]^T is the HNF of[B^T|a_t]^T. 

    @param B (MyMatrix): a square integer matrix of of dimension.
    @param H_B (MyMatrix): the HNF of B.
    @param a_t (MyMatrix): the row vector we want to add  
    @return (MyMatrix): x.
    """
    n = B.getRows()
    D = None
    if H_B.getRows() == H_B.getCols():
        D = abs(H_B.getDiagProd()) # compute determinant of B bzw. H_B
    else:
        t = H_B.getTranspose() @ H_B
        D = round(sqrt(t.getDet()))
    product = 1
    chosen_primes = []
    p = 2**31

    M = B.getLargestEntry()
    for e in a_t.getList():
        if abs(e) > M:
            M = abs(e)
    bound = n**(n+1)*M**(2*n+1)

    while product <= 2 * bound:
        p = nextprime(p)
        if not D % p == 0:
            chosen_primes.append(p)
            product *= p
    x_vectors = []
    #LGS
    for i in range(len(chosen_primes)):
        y = B.getTranspose().linear_solve(a_t, mod_p=chosen_primes[i])
        x = H_B.getTranspose().mulMod(y, chosen_primes[i])
        x_vectors.append(x.getList())
    x_res = []
    #CRT
    for i in range(len(x_vectors[0])):
        coords = []
        for j in range(len(x_vectors)):
            if isinstance(x_vectors[j][i], list):
                coords.append(x_vectors[j][i][0])
            else:
                coords.append(x_vectors[j][i])
        result = crt(coords, chosen_primes)

        if result > product // 2:
            result -= product
        x_res.append(result)

    res = MyMatrix(x_res)
    return res

def addColumn(A: MyMatrix, b: list) -> MyMatrix:
    """
    Computes the HNF H of [A|b]. 

    @param A (MyMatrix): an integer matrix of of dimension nxn-1 or nxn.
    @param b (list): the (column-)vector of dimension n we want to add.  
    @return (MyMatrix): the HNF of [A|b].
    """
    n = A.getRows()
    H_all = A.copy()
    A.addColumn(b)
    d_ = abs(A.getDet())
    c_ = [0] * n
    c_[n - 1] = d_
    m = [0] * n
    if H_all.getCols() < H_all.getRows():
        if d_ == 0:
            return H_all
        # fill matrix to be square that generate the same lattice
        H_all.addColumn(c_)
        m[n - 1] = abs(d_)
    else:
        m[n - 1] = abs(H_all.getDet())

    for i in range(n - 1, 0, -1):
        m[i - 1] = abs(m[i] * H_all[i, i])
        assert m[i - 1] != 0, "lol how is that possible?"

    for j in range(1, n + 1):
        g, r, y = eea(H_all[j, j], b[j - 1])
        bg = -b[j - 1] // g
        hg = H_all[j, j] // g
        for i in range(j, n + 1):
            b_i_save = b[i - 1]
            b[i - 1] = (H_all[i, j] * bg + b[i - 1] * hg) % m[i - 1]
            H_all[i, j] = (r * H_all[i, j] + y * b_i_save) % m[i - 1]
        if H_all[j, j] == 0:
            H_all[j, j] = m[j - 1]
        # reduce left side of column j
        for c in range(1, j):
            q = H_all[j, c] // H_all[j, j]
            for r in range(j, n + 1):  # start at row j -> above is same as subtracting q* zero!
                H_all[r, c] = (H_all[r, c] - q * H_all[r, j]) % m[r - 1]
        # reduce right side as shown in paper
        for k in range(j + 1, n + 1):
            q = H_all[k, j] // H_all[k, k]
            for l in range(k, n + 1):
                H_all[l, j] = (H_all[l, j] - q * H_all[l, k]) % m[l - 1]

    return H_all


def hnf_fast(A: MyMatrix) -> MyMatrix:
    """
    hnf_fast() computes the HNF of a square nonsingular matrix using the Heuristic Algorithm

    @param A (MyMatrix): a nonsingular Matrix A.
    @return (MyMatrix): the HNF of A.
    """
    input_matrix = A.copy()
    n = input_matrix.getCols()
    assert n > 2, "n must be greater than 2 for the heursitic version"
    c = MyMatrix(input_matrix.getColumn(n - 1)[:-1])
    d = MyMatrix(input_matrix.getColumn(n)[:-1])
    b_t = MyMatrix(input_matrix.getRow(n)[:-2])
    B = input_matrix.getPrincipalSubmatrix(n - 1)
    B.removeColumn(n - 1)

    B.addColumn(c.getList())
    d1 = B.getDet()
    B.removeColumn(n-1)
    B.addColumn(d.getList())
    d2 = B.getDet()
    B.removeColumn(n - 1)
    g, k, l = eea(d1, d2) #TODO: Must choose d1 not abs(d1) here etc. ! det() is not multilinear on absolute values!
    c.scalarMultiply(k)
    d.scalarMultiply(l)
    c.add(d)  # c = kc+ld
    B.addColumn(c.getList())
    H = MyMatrix.getIdentity(B.getRows())
    g = abs(g)
    if g > 1:
        H = hnf_mod_D(B, g)

    b_t.addColumn([k * input_matrix[n, n - 1] + l * input_matrix[n, n]])
    x_t = addRow(B, H, b_t)
    H_prime = H.addRow(x_t)

    H_prime = addColumn(H_prime, input_matrix.getColumn(n - 1))
    H_prime = addColumn(H_prime, input_matrix.getColumn(n))

    return H_prime



def hnf(A: MyMatrix) -> MyMatrix:
    """
    hnf() computes the HNF of a square matrix, where every principal minor is nonzero. This must be guaranteed by the caller. 

    @param A (MyMatrix): a nonsingular Matrix A, where all principal minors are nonzero.
    @return (MyMatrix): the HNF of A.
    """
    input_matrix = A.copy()
    # for the case n = 1
    if input_matrix[1, 1] < 0:
        for i in range(input_matrix.getRows()):
            input_matrix[i + 1, 1] *= -1

    # return if it is the base case
    n = input_matrix.getCols()
    if n == 1:
        return input_matrix

    H = input_matrix.getPrincipalSubmatrix(1)
    for i in range(2, n + 1):
        x = addRow(input_matrix.getPrincipalSubmatrix(i - 1), H, input_matrix.get_a_t((i - 1)))
        new_matrix = H.addRow(x)
        H = addColumn(new_matrix, input_matrix.get_a_column(i))
    return H



def hnf_universal(M: MyMatrix) -> MyMatrix:
    """
    Computes the HNF of an arbitrary maitrx M in Mat(m,n,Z).
    For matrices with 2 rows, it uses the Linear Space Algorithm (hnf()),
    for all other matrices it uses the Heuristic Algorihm (hnf_fast()).

    @param M (MyMatrix): an arbitrary matrix M
    @return (MyMatrix): the HNF of M
    """
    if M.getRows() == M.getCols():
        if M.getDet() != 0:
            M.transformToAllPrincipalMinorsNonZero()
            if M.getRows() > 2:
                return hnf_fast(M)
            else:
                return hnf(M)

    M_copy = M.copy()
    pivots_cols = M.getProbablePivots()
    k = 0
    for i in range(1, M.getCols()+1):
        if i not in pivots_cols:
            M.removeColumn(i-k)
            k+= 1
    pivots_rows = M.getTranspose().getProbablePivots()
    M_after_columns = M.copy()
    k = 0
    for i in range(1, M.getRows()+1):
        if i not in pivots_rows:
            M.removeRow(i-k)
            k += 1
    M.transformToAllPrincipalMinorsNonZero()
    H = None
    if M.getRows() > 2:
        H = hnf_fast(M)
    else:
        H = hnf(M)
    for i in range(1, M_after_columns.getRows()+1):
        if i not in pivots_rows:
            x = addRow(H, H, MyMatrix(M_after_columns.getRow(i)))
            H.addRow(x)
    for i in range(1, M_copy.getCols() + 1):
        if i not in pivots_cols:
            H = addColumn(H, M_copy.getColumn(i))

    return H


if __name__ == "__main__":
    test = MyMatrix([[30, 45, -49, -38, -36],
                      [-26, -29, -16, -50, 25],
                      [38, -23, -43, 9, -47],
                      [-17, 7, -48, 24, 13],
                      [2, 1, 44, 21, -26]])


    import time
    time1 = time.time_ns()
    h1 = hnf(test)
    time2 = time.time_ns()

    print("Execution time: ", (time2 - time1) / (10 ** 9), " seconds")
    print('\033[94m' + "[INPUT]\n " + '\033[92m B\033[0m', " =\n", t1)
    if h1.isHNF() and abs(h1.getDet()) == abs(t1.getDet()):
        print('\033[94m' + "[RESULT]\n " + '\033[92m H\033[0m', " =\n", h1)
    else:
        print("det(H) = ", h1.getDet(), ", det(B) = ", t1.getDet())
        print('\033[91m' + "[ERROR RESULT]\n " + 'H\033[0m', " =\n", h1)

