from builtins import isinstance
from Utility import *
from sympy import sqrt
import numpy as np

class MyMatrix:
    def __init__(self, m: list):
        if type(m) is not type([]):
            exit("You must provide a parameter of type list")
        elif len(m) > 0:
            if type(m[0]) is not type([]):
                m = [m]  # un-flatten()
        else:
            m = [[]]

        self.__matrix = m
        self.__rows = len(m)
        self.__columns = len(m[0])

    def __str__(self):
        res = ""
        for r in range(self.__rows):
            res += str(self.__matrix[r]) + "\n"
        return res

    def __matmul__(self, other):
        result_rows = self.__rows
        result_cols = other.getCols()
        res = []
        for i in range(result_rows):
            res.append([])
            for j in range(result_cols):
                res[i].append(0)
                for k in range(self.__columns):
                    res[i][j] += self.__matrix[i][k] * other[k + 1, j + 1]
        return MyMatrix(res)

    def __getitem__(self, item):
        x, y = item
        if x < 1 or y < 1:
            raise IndexError
        return self.__matrix[x - 1][y - 1]

    def __setitem__(self, key, value):
        x, y = key
        self.__matrix[x - 1][y - 1] = value
        self.__matrix[x - 1][y - 1] = value

    def __eq__(self, other):
        if not isinstance(other, MyMatrix):
            exit("cannot compare MyMatrix to " + str(type(other)))
        if (other.getCols() != self.__columns) or (other.getRows() != self.__rows):
            return False
        for i in range(self.__rows):
            for j in range(self.__columns):
                if self.__matrix[i][j] != other[i + 1, j + 1]:
                    return False
        return True

    def getColumn(self, i):
        c = []
        for r in range(self.__rows):
            c.append(self.__matrix[r][i - 1])
        return c

    def getRow(self, i):
        return self.__matrix[i - 1]

    def getCols(self):
        return self.__columns

    def getRows(self):
        return self.__rows

    def getDet(self):
        #import numpy as np
        #from numpy.linalg import det
        #m = np.array(self.__matrix)
        #d = round(det(m))
        #return d

        M = self.__matrix
        M = [row[:] for row in M]  # make a copy to keep original M unmodified
        N, sign, prev = len(M), 1, 1
        for i in range(N - 1):
            if M[i][i] == 0:  # swap with another row having nonzero i's elem
                swapto = next((j for j in range(i + 1, N) if M[j][i] != 0), None)
                if swapto is None:
                    return 0  # all M[*][i] are zero => zero determinant
                M[i], M[swapto], sign = M[swapto], M[i], -sign
            for j in range(i + 1, N):
                for k in range(i + 1, N):
                    assert (M[j][k] * M[i][i] - M[j][i] * M[i][k]) % prev == 0
                    M[j][k] = (M[j][k] * M[i][i] - M[j][i] * M[i][k]) // prev
            prev = M[i][i]
        return sign * M[-1][-1]

    def getTranspose(self):
        r = []
        for i in range(self.__columns):
            new_row = []
            for j in range(self.__rows):
                new_row.append(self.__matrix[j][i])
            r.append(new_row)
        return MyMatrix(r)

    def ensureIntegers(self, mod_p=None):
        if mod_p is None:
            for i in range(self.__rows):
                for j in range(self.__columns):
                    self.__matrix[i][j] = round(self.__matrix[i][j])
        else:
            for i in range(self.__rows):
                for j in range(self.__columns):
                    self.__matrix[i][j] = (round(self.__matrix[i][j])) % mod_p

    @staticmethod
    def getIdentity(n):
        res = []
        for i in range(n):
            new_row = []
            for j in range(n):
                if i == j:
                    new_row.append(1)
                else:
                    new_row.append(0)
            res.append(new_row)
        return MyMatrix(res)

    def removeColumn(self, idx):
        for i in range(self.__rows):
            del self.__matrix[i][idx - 1]

        self.__columns -= 1

    def removeRow(self, idx):
        del self.__matrix[idx-1]
        self.__rows -= 1


    def getInverse(self, unimodular=False):
        """
          Only for debugging/testing purposes! uses floating points and thus fails for multi precision arithmetic!
        """
        from mpmath import matrix as MpMatrix
        m = MpMatrix(self.__matrix)
        res = m ** -1
        res = res.tolist()  # convert mpf datatype to native datatype
        for i in range(len(res)):
            for j in range(len(res[0])):
                if unimodular:
                    res[i][j] = int(res[i][j])
                else:
                    res[i][j] = float(res[i][j])
        res = MyMatrix(res)
        return res

    def scalarMultiply(self, a):
        for i in range(self.__rows):
            for j in range(self.__columns):
                self.__matrix[i][j] *= a

    def add(self, v: MyMatrix):
        for i in range(v.getRows()):
            for j in range(v.getCols()):
                self.__matrix[i][j] += v[i + 1, j + 1]

    def addRow(self, r: list):
        if type(r) == type(self):
            l = r.getRow(1)
            r = l
        if len(r) != self.__columns:
            exit("MyMatrix.addRow: Dimension mismatch")
        self.__matrix.append(r)
        self.__rows += 1
        return self

    def getPrincipalSubmatrix(self, m):
        res = []
        for i in range(m):
            r = []
            for j in range(m):
                r.append(self.__matrix[i][j])
            res.append(r)
        return MyMatrix(res)

    def addColumn(self, c: list):
        if len(c) != self.__rows:
            exit("MyMatrix.addColumn: Dimension mismatch")
        for i in range(self.__rows):
            self.__matrix[i].append(c[i])
        self.__columns += 1
        return self

    def get_a_t(self, idx):
        """
        returns the vector a_t(i) as defined in the paper.
        It is used as the input to AddRow.
        """
        r = self.getRow(idx + 1)
        return MyMatrix(r[:idx])

    def get_a_column(self, idx):
        """
        Column equivalent to get_a_t(idx).
        It returns the input c(i) to AddColumn.
        """
        c = self.getColumn(idx)
        return c[:idx]

    def isHNF(self):
        for i in range(self.__rows):
            if self.__matrix[i][i] <= 0:
                return False
            for j in range(self.__columns):
                if self.__matrix[i][j] < 0:
                    return False
                if self.__matrix[i][j] > self.__matrix[i][i]:
                    return False
                if j > i and self.__matrix[i][j] != 0:
                    return False
        return True

    def copy(self):
        """
        return deep copy.
        """
        res = []
        for i in range(self.__rows):
            r = []
            for j in range(self.__columns):
                r.append(self.__matrix[i][j])
            res.append(r)
        #res = self.__matrix.copy()
        return MyMatrix(res)

    def getNextNonZeroColumnIndex(self, c):
        for i in range(c+1, self.__columns):
           if self.__matrix[c-1][i-1] != 0:
               return i
        return None

    @staticmethod
    def get_e_i(n, index, new_value=None):
        t = [0]*n
        if new_value is None:
            t[index] = 1
        else:
            t[index] = new_value
        return MyMatrix(t)
    def invertColumn(self, c):
        for i in range(self.__rows):
            self.__matrix[i][c-1] = -self.__matrix[i][c-1]


    def generateZeroRowMod(self, row, d):
        idx = self.__columns // 2 + row - 1
        self.__matrix[row-1][idx] = d[row]
        for col in range(row+1, self.__columns//2 + row + 1):
            self.generateZero(row, col)
        for c in range(row, self.__columns//2 + row + 1):
            for r in range(row, self.__rows+1):
                self.__matrix[r - 1][c - 1] %= d[r]
        if self.__matrix[row-1][row-1] == 0:
            self.__matrix[row-1][row-1] = d[row]

    
    def generateZero(self, row, col, d=None):
        """
        implementation of the matrix multiplication with U_{rc}
        """
        if self.__matrix[row-1][col-1] == 0:
            return
        j = row
        g, r, y = eea(self.__matrix[j-1][j-1], self.__matrix[j-1][col-1])
        if d is not None and g == 0:
            g = d[j]
        bg = -self.__matrix[j-1][col-1] // g
        hg = self.__matrix[j-1][j-1] // g
        for i in range(j, self.__rows + 1):
            b_i_save = self.__matrix[i-1][col-1]
            self.__matrix[i-1][col-1] = (self.__matrix[i-1][j-1] * bg + self.__matrix[i-1][col-1] * hg)
            self.__matrix[i-1][j-1] = (r * self.__matrix[i-1][j-1] + y * b_i_save)
            if d is not None:
                self.__matrix[i - 1][col - 1] %= d[i]
                self.__matrix[i - 1][j - 1] %= d[i]
        if d is not None and 1 <= col <= self.__rows and self.__matrix[col-1][col-1] == 0:
            self.__matrix[col-1][col-1] = d[col]

    def reduceColumnMod(self, c, end, d=None):
        for r in range(c, end):
            q = self.__matrix[r][c-1] // self.__matrix[r][r]
            for k in range(r, self.__rows):
                self.__matrix[k][c-1] -= q * self.__matrix[k][r]
                if d is not None:
                    self.__matrix[k][c-1] %= d[k+1]

    def reduceColumn(self, c, d=None):
        for r in range(c, self.__rows):
            if d is not None:
                self.__matrix[r][r] %= d[r+1]
                if self.__matrix[r][r] == 0:
                    self.__matrix[r][r] = d[r+1]
            q = self.__matrix[r][c-1] // self.__matrix[r][r]
            for k in range(r, self.__rows):
                self.__matrix[k][c-1] -= q * self.__matrix[k][r]
                if d is not None:
                    self.__matrix[k][c-1] %= d[k+1]
            if self.__matrix[c-1][c-1] == 0:
                self.__matrix[c-1][-1] = d[c]


    def transformToAllPrincipalMinorsNonZero(self):
        for i in range(1, self.__rows+1):
            d = self.getPrincipalSubmatrix(i).getDet()
            j = i
            while j < self.__rows and d == 0:
                j += 1
                self.switchColumns(i, j)
                d = self.getPrincipalSubmatrix(i).getDet()
            if j > self.__rows:
                print("A is singular!")
                exit("ERROR in transformToAllPrincipalMinorsNonZero")

    def getLargestEntry(self):
        M = -1
        for i in range(self.__rows):
            for j in range(self.__columns):
                if (x := abs(self.__matrix[i][j])) > M:
                     M = x
        return M


    def switchColumns(self, a: int, b: int):
        for i in range(0, self.__rows):
            tmp = self.__matrix[i][a - 1]
            self.__matrix[i][a - 1] = self.__matrix[i][b - 1]
            self.__matrix[i][b - 1] = tmp

    def getList(self):
        if len(self.__matrix) == 1:
            return list(self.__matrix[0])
        return list(self.__matrix)

    def mulMod(self, other: MyMatrix, modulus: int):
        result_rows = self.__rows
        result_cols = other.getCols()
        res = []
        for i in range(result_rows):
            res.append([])
            for j in range(result_cols):
                res[i].append(0)
                for k in range(self.__columns):
                    res[i][j] += (self.__matrix[i][k]) * (other[k + 1, j + 1]) % modulus
                    res[i][j] %= modulus
        return MyMatrix(res)

    def scalarMultiply(self, a: int):
        for i in range(self.__rows):
            for j in range(self.__columns):
                self.__matrix[i][j] *= a

    def getDiagProd(self):
        p = 1
        for i in range(self.__rows):
            p *= self.__matrix[i][i]
        return p

    def add(self, v: MyMatrix):
        for i in range(v.getRows()):
            for j in range(v.getCols()):
                self.__matrix[i][j] += v[i + 1, j + 1]

    def getMaxColumnLength(self):
        m = -1
        for j in range(self.__columns):
            res = 0
            for e in self.getColumn(j + 1):
                res += e ** 2
            norm = sqrt(res)
            if norm > m:
                m = norm
        return m

    def hasFullRank(self, mod_p=None):
        if self.__columns != self.__rows:
            return False
        try:
            if mod_p is None:
                if self.getDet() == 0:
                    return False
            else:
                if self.getDet() % mod_p == 0:
                    return False
        except ZeroDivisionError:
            return False

        return True

    def getProbablePivots(self):
        """
        return the column indices of the pivot elements (probablistic) by using Gaussian elimination over Z_p for random p.
        """
        from random import randint
        from sympy import nextprime

        r = randint(10007, 46000)
        p = nextprime(r)
        m = self.getCols()
        n = self.getRows()
        b_list = [0]*n
        m_list = self.copy()
        m_list = m_list.getList()
        if isinstance(m_list[0], int):
            m_list = [m_list]
        for i in range(n):
            for j in range(m):
                m_list[i][j] %= p
        if not isinstance(m_list[0], list):
            m_list = [m_list]
        # ensure all diagonal elements are nonzero
        for i in range(min(m,n)):
                if m_list[i][i] == 0:
                    for r in range(i, n):
                        if m_list[r][i] != 0:
                            # switch rows
                            row_i = m_list[i]
                            row_r = m_list[r]
                            m_list[i] = row_r
                            m_list[r] = row_i
                            # switchRows(m_list, i, r)
                            break
        # Elemination phase
        for col in range(0, min(m,n)):
            if m_list[col][col] == 0:
                continue
            # make upper triangular
            for row in range(col + 1, min(m,n)):
                multiplier = (-m_list[row][col]) / m_list[col][col]
                for i in range(m):
                    m_list[row][i] = m_list[row][i] + multiplier * m_list[col][i]
                b_list[row] = b_list[row] + multiplier * b_list[col]
            # make lower triangluar => only diagonal elements
            for row in range(0, col):
                multiplier = (-m_list[row][col]) / m_list[col][col]
                for i in range(m):
                    m_list[row][i] = m_list[row][i] + multiplier * m_list[col][i]
                b_list[row] = b_list[row] + multiplier * b_list[col]
            # normalize diagonal elements
        res = []
        for r in range(min(m,n)):
            for i in range(min(m,n)):
                if m_list[r][i] != 0:
                    res.append(i+1)
                    break
        return res
    

    def linear_solve(self, b_: MyMatrix, mod_p=None):
        """
        solve Ax=b
        """
        if not isinstance(b_, MyMatrix):
            exit("Need type MyMatrix to solve gauss")
        b = b_.copy()
        if b.getRows() > 1 and b.getCols() == 1:
            b = b.getTranspose()
            # -> b has form [[b1, b2, ... ,bn]]
        n = b.getCols()
        # assert self.hasFullRank()
        if mod_p is not None:
            b.ensureIntegers(mod_p=mod_p)
            b_list = b.getList()
            m_list = self.copy()
            pivots = m_list.getProbablePivots()
            k = 0
            for i in range(1, m_list.getCols()+1):
                if i not in pivots:
                    m_list.removeColumn(i-k)
                    k += 1
            m_list.ensureIntegers(mod_p)
            m_list = m_list.getList()
            if not isinstance(m_list[0], list):
                m_list = [m_list]
            # guarantee all diagonal elements are nonzero

            # Elemination phase
            for col in range(0, n):
                # switch rows so that diagonal element is nonzero
                if m_list[col][col] == 0:
                    for r in range(col, n):
                        if m_list[r][col] != 0:
                            # switch rows
                            row_i = m_list[col]
                            row_r = m_list[r]
                            m_list[col] = row_r
                            m_list[r] = row_i
                            row_i = b_list[col]
                            row_r = b_list[r]
                            b_list[col] = row_r
                            b_list[r] = row_i
                            # switchRows(m_list, i, r)
                            break
                # make upper triangular
                for row in range(col + 1, n):
                    multiplier = ((-m_list[row][col]) * modInverse(m_list[col][col], mod_p)) % mod_p
                    for i in range(n):
                        m_list[row][i] = (m_list[row][i] + multiplier * m_list[col][i]) % mod_p
                    b_list[row] = (b_list[row] + multiplier * b_list[col]) % mod_p
                # make lower triangluar => only diagonal elements
                for row in range(0, col):
                    multiplier = ((-m_list[row][col]) * modInverse(m_list[col][col], mod_p)) % mod_p
                    for i in range(n):
                        m_list[row][i] = (m_list[row][i] + multiplier * m_list[col][i]) % mod_p
                    b_list[row] = (b_list[row] + multiplier * b_list[col]) % mod_p
                # normalize diagonal elements
            for i in range(n):
                m = modInverse(m_list[i][i], mod_p)
                m_list[i][i] = 1
                b_list[i] = (b_list[i] * m) % mod_p
            return MyMatrix(b_list).getTranspose()
        else:
            b_list = b.getList()
            m_list = self.copy()
            m_list = m_list.getList()
            if not isinstance(m_list[0], list):
                m_list = [m_list]
            # ensure all diagonal elements are nonzero
            for i in range(n):
                if m_list[i][i] == 0:
                    for r in range(i, n):
                        if m_list[r][i] != 0:
                            # switch rows
                            row_i = m_list[i]
                            row_r = m_list[r]
                            m_list[i] = row_r
                            m_list[r] = row_i
                            # switchRows(m_list, i, r)
                            break
            # Elemination phase
            for col in range(0, n):
                # make upper triangular
                for row in range(col + 1, n):
                    multiplier = (-m_list[row][col]) / m_list[col][col]
                    for i in range(n):
                        m_list[row][i] = m_list[row][i] + multiplier * m_list[col][i]
                    b_list[row] = b_list[row] + multiplier * b_list[col]
                # make lower triangluar => only diagonal elements
                for row in range(0, col):
                    multiplier = (-m_list[row][col]) / m_list[col][col]
                    for i in range(n):
                        m_list[row][i] = m_list[row][i] + multiplier * m_list[col][i]
                    b_list[row] = b_list[row] + multiplier * b_list[col]
                # normalize diagonal elements
            for i in range(n):
                m = 1 / m_list[i][i]
                m_list[i][i] = 1
                b_list[i] = b_list[i] * m
            return MyMatrix(b_list).getTranspose()

