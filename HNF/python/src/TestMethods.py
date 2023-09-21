from HNF import *
from MyMatrix import MyMatrix
import unittest


class TestMethods(unittest.TestCase):

    # parameter to test the HNF-methods:
    max = 500 # number of repititions
    n = 25 # dimension of matrix
    a = -2**10 #minimal entry
    b = 2**10 #maximal entry

    def test_eea(self):
        for i in range(-10, 30):
            for j in range(-20, 40):
                g, x, y = eea(i, j)
                self.assertEqual(g, x * i + y * j)
        self.assertTrue(eea(0, 10)[0] == eea(10, 0)[0] == 10)

    def test_addRowToMatrix(self):
        v1 = [1, 2, 3]
        M1 = MyMatrix([[2, 34, 101],
                       [0, 1, 44],
                       [0, 17, 1]])
        M2 = M1.copy()

        M1.addRow(v1)
        self.assertTrue(M1 == MyMatrix([[2, 34, 101],
                                        [0, 1, 44],
                                        [0, 17, 1],
                                        [1, 2, 3]]))

    def test_addColumnToMatrix(self):
        v1 = [1, 2, 3]
        M = MyMatrix([[34, 101],
                      [1, 44],
                      [17, 1]])
        M.addColumn(v1)
        self.assertTrue(M == MyMatrix([[34, 101, 1],
                                       [1, 44, 2],
                                       [17, 1, 3]]))

    def test_getPrincipalMinor(self):
        M = MyMatrix([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        m1 = M.getPrincipalSubmatrix(1)
        m2 = M.getPrincipalSubmatrix(2)
        m3 = M.getPrincipalSubmatrix(3)
        self.assertTrue(m1 == MyMatrix([1]))
        self.assertTrue(m2 == MyMatrix(([[1, 2], [4, 5]])))
        self.assertTrue(m3 == MyMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    def test_removeColumn(self):
        M = MyMatrix([[1, 2, 3], [4, 5, 6]])
        M.removeColumn(1)
        self.assertTrue(M == MyMatrix([[2, 3], [5, 6]]))
        M = MyMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        M.removeColumn(3)
        self.assertTrue(M == MyMatrix([[1, 2], [4, 5], [7, 8]]))

    def test_getIdentity(self):
        I = MyMatrix.getIdentity(1)
        self.assertTrue(I == MyMatrix([1]))
        I = MyMatrix.getIdentity(3)
        self.assertTrue(I == MyMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_get_a_t(self):
        M = MyMatrix([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        a0 = M.get_a_t(1)
        a1 = M.get_a_t(2)
        self.assertTrue(a0 == MyMatrix([4]))
        self.assertTrue(a1 == MyMatrix([7, 8]))

    def test_MatMul(self):
        A = MyMatrix([[1, 2, 3]])
        B = MyMatrix([[1], [2], [3]])
        self.assertTrue(A @ B == MyMatrix([14]))
        A = MyMatrix([[1, 2, 3], [0, 4, 5]])
        B = MyMatrix([[1, 2, 4], [2, 4, 5], [3, 5, 6]])
        self.assertTrue((A @ B) == MyMatrix([[14, 25, 32], [23, 41, 50]]))
        A = MyMatrix([[1, 0], [0, 1]])
        B = MyMatrix([[0xFFFFFFFFFFFFFFFFFFFF ** 0xF, 0xFFFFFFFAAAAAAAAAAEEEEEEEEEEEEE ** 0xF],
                      [0xAAAAAAAAAAAFFFFFFFFFFFFFEEEEEEEEEEEE ** 0xEEA, 0xFFFFFFFFFFFFEEEEEEEEEEEBBBBBB ** 0xFFF]])
        self.assertTrue(A @ B == B)

    def test_mulMod(self):
        A = MyMatrix([[56, 76, 34], [1, 3, 55], [4, 6, 684]])
        r = A @ A
        p = 13
        r.ensureIntegers(p)
        r2 = A.mulMod(A, p)
        self.assertEqual(r, r2)
        self.assertEqual(A, MyMatrix([[56, 76, 34], [1, 3, 55], [4, 6, 684]]))


    def test_getList(self):
        M = MyMatrix([[1, 2], [3, 3], [4, 4], [5, 5]])
        self.assertTrue(M.getList() == [[1, 2], [3, 3], [4, 4], [5, 5]])
        M = MyMatrix([[1, 2, 3]])
        self.assertTrue(M.getList() == [1, 2, 3])
        M = MyMatrix([[1], [2], [3]])
        self.assertTrue(M.getList() == [[1], [2], [3]])

    def test_get_a_column(self):
        M = MyMatrix([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        c0 = M.get_a_column(1)
        c1 = M.get_a_column(2)
        c2 = M.get_a_column(3)
        self.assertTrue(c0 == [1])
        self.assertTrue(c1 == [2, 5])
        self.assertTrue(c2 == [3, 6, 9])

    def test_linear_solve(self):
        M = MyMatrix([[2, 1], [1, 1]])
        b = MyMatrix([[1, 0]])
        r = M.linear_solve(b, mod_p=2)
        self.assertTrue(r == MyMatrix([1, 1]).getTranspose())

    def test_hasFullRank(self):
        M = MyMatrix([[1, 0], [0, 1]])
        self.assertTrue(M.hasFullRank())
        M = MyMatrix([[1, 0], [1, 1]])
        self.assertTrue(M.hasFullRank())
        M = MyMatrix([[1, 1], [1, 1]])
        self.assertFalse(M.hasFullRank())
        M = MyMatrix([[1, 0], [1, 0]])
        self.assertFalse(M.hasFullRank())
        M = MyMatrix([[5, 0], [1, 5]])
        self.assertTrue(M.hasFullRank())
        self.assertFalse(M.hasFullRank(mod_p=5))
        self.assertTrue(M.hasFullRank(mod_p=7))

    def test_modInverse(self):
        self.assertEqual(modInverse(1, 2), 1)
        self.assertEqual(modInverse(-1, 2), 1)
        self.assertEqual(modInverse(-2, 2), -1)
        self.assertEqual(modInverse(5, 15), -1)
        self.assertEqual(modInverse(5, 3), 2)

    def test_inverseMatrix(self):
        A = MyMatrix([[1, 2], [1, 1]])
        self.assertTrue(A @ A.getInverse() == MyMatrix.getIdentity(2))
        A = MyMatrix([[1, 2, 3], [4, 5, 6], [69, 8, 9]])
        res = A @ A.getInverse()
        res.ensureIntegers()
        self.assertTrue(res == MyMatrix.getIdentity(3))

    def test_isHNF(self):
        A1 = MyMatrix([1])
        self.assertTrue(A1.isHNF())
        A2 = MyMatrix([[1, 0],
                       [0, 3]])
        self.assertTrue(A2.isHNF())
        A3 = MyMatrix([[1, 1],
                       [0, 3]])
        self.assertFalse(A3.isHNF())
        A4 = MyMatrix([[0, 0],
                       [0, 3]])
        self.assertFalse(A4.isHNF())
        A5 = MyMatrix([[1, 0, 0],
                       [2, 3, 0]])
        self.assertTrue(A5.isHNF())
        A6 = MyMatrix([[1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 1, 3, 0],
                       [616, 0, 882, 330, 1232]])
        self.assertTrue(A6.isHNF())
        A7 = MyMatrix([[1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 1, 3, 0],
                       [616, 0, 882, 330, 1232]])
        self.assertFalse(A7.isHNF())
        A8 = MyMatrix([[1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 1, 3, 0],
                       [616, 1233, 882, 330, 1232]])
        self.assertFalse(A8.isHNF())

    def test_all_hnf_equal(self):
        import numpy as np
        for _ in range(50):
            A = MyMatrix(
                np.random.randint(low=-50, high=50, size=(7, 7)).tolist())
            c = 0
            for j in range(1, 7):
                if A.getPrincipalSubmatrix(j).getDet() == 0:
                    c = 1

            if c == 1:
                continue

            H1 = hnf(A)
            H2 = hnf_basic(A)
            H3 = hnf_mod_D(A)
            H4 = hnf_mod_D(A)
            self.assertEqual(H1, H2)
            self.assertEqual(H2, H3)
            self.assertEqual(H3, H4)


    def test_hnf_mod_D(self):
        import numpy as np
        A = MyMatrix.getIdentity(10)
        H = hnf_mod_D(A)
        self.assertTrue(H.isHNF() and abs(H.getDet()) == abs(A.getDet()),
                        "A=\n" + str(A) + "\ndet(A)=" + str(A.getDet()) + "\nH=\n" + str(H) + "\ndet(H)=" + str(
                            H.getDet()))
        for i in range(TestMethods.max):
            # print(i)
            A = MyMatrix(np.random.randint(low=TestMethods.a, high=TestMethods.b, size=(TestMethods.n,TestMethods.n)).tolist())
            c = 0
            for j in range(1, TestMethods.n + 1):
                if A.getPrincipalSubmatrix(j).getDet() == 0:
                    c = 1

            if c == 1:
                continue
            # print("start")
            import time
            time1 = time.time_ns()
            #COPY = A.copy()
            H = hnf_mod_D(A)
            #self.assertEqual(A, COPY)
            time2 = time.time_ns()
            p = int(i / TestMethods.max * 100)
            s = "[" + "#" + "#" * (int(p / 100 * 30)) + " " * (int((1 - p / 100) * 30)) + "] " + str(p) + "%"
            print("\r" + '\033[92m' + s, end="", flush=True)
            print(" - " + '\033[0m' + "Execution time: ", (time2 - time1) / (10 ** 9), " seconds", end="", flush=True)
            # U = A.getInverse() @ H    # int too large to convert to float
            # U.ensureIntegers()
            # H_prime = A @ U
            # H_prime.ensureIntegers()
            self.assertTrue(H.isHNF() and abs(H.getDet()) == abs(A.getDet()),
                            "A=\n" + str(A) + "\ndet(A)=" + str(A.getDet()) + "\nH=\n" + str(H) + "\ndet(H)=" + str(
                                H.getDet()) + "\ncorrect=\n" + str(hnf(A)))



    def test_hnf_basic(self):
        import numpy as np
        A = MyMatrix.getIdentity(10)
        H = hnf_basic(A)
        self.assertTrue(H.isHNF() and abs(H.getDet()) == abs(A.getDet()),
                        "A=\n" + str(A) + "\ndet(A)=" + str(A.getDet()) + "\nH=\n" + str(H) + "\ndet(H)=" + str(
                            H.getDet()))
        for i in range(TestMethods.max):
            # print(i)
            A = MyMatrix(np.random.randint(low=TestMethods.a, high=TestMethods.b, size=(TestMethods.n,TestMethods.n)).tolist())
            c = 0
            for j in range(1, TestMethods.n + 1):
                if A.getPrincipalSubmatrix(j).getDet() == 0:
                    c = 1

            if c == 1:
                continue
            #print("start")
            import time
            time1 = time.time_ns()
            #COPY = A.copy()
            H = hnf_basic(A)
            #self.assertEqual(COPY, A)
            time2 = time.time_ns()
            p = int(i / TestMethods.max * 100)
            s = "[" + "#" + "#" * (int(p/100 * 30)) + " " * (int((1 - p/100) * 30)) + "] " + str(p) + "%"
            print("\r" + '\033[92m' + s, end="", flush=True)
            print(" - " + '\033[0m' + "Execution time: ", (time2 - time1) / (10 ** 9), " seconds", end="", flush=True)
            # U = A.getInverse() @ H    # int too large to convert to float
            # U.ensureIntegers()
            # H_prime = A @ U
            # H_prime.ensureIntegers()
            self.assertTrue(H.isHNF() and abs(H.getDet()) == abs(A.getDet()),
                            "A=\n" + str(A) + "\ndet(A)=" + str(A.getDet()) + "\nH=\n" + str(H) + "\ndet(H)=" + str(
                                H.getDet()) + "\ncorrect=\n" + str(hnf(A)))

    def test_hnf(self):
        import numpy as np
        A = MyMatrix.getIdentity(10)
        H = hnf(A)
        self.assertTrue(H.isHNF() and abs(H.getDet()) == abs(A.getDet()),
                        "A=\n" + str(A) + "\ndet(A)=" + str(A.getDet()) + "\nH=\n" + str(H) + "\ndet(H)=" + str(
                            H.getDet()))
        for i in range(TestMethods.max):
            # print(i)
            A = MyMatrix(np.random.randint(low=TestMethods.a, high=TestMethods.b, size=(TestMethods.n,TestMethods.n)).tolist())
            c = 0
            for j in range(1, TestMethods.n + 1):
                if A.getPrincipalSubmatrix(j).getDet() == 0:
                    c = 1

            if c == 1:
                continue
            #print("start")
            import time
            time1 = time.time_ns()
            H = hnf(A)
            time2 = time.time_ns()
            p = int(i / TestMethods.max * 100)
            s = "[" + "#" + "#" * (int(p/100 * 30)) + " " * (int((1 - p/100) * 30)) + "] " + str(p) + "%"
            print("\r" + '\033[92m' + s, end="", flush=True)
            print(" - " + '\033[0m' + "Execution time: ", (time2 - time1) / (10 ** 9), " seconds", end="", flush=True)
            # U = A.getInverse() @ H    # int too large to convert to float
            # U.ensureIntegers()
            # H_prime = A @ U
            # H_prime.ensureIntegers()
            self.assertTrue(H.isHNF() and abs(H.getDet()) == abs(A.getDet()),
                            "A=\n" + str(A) + "\ndet(A)=" + str(A.getDet()) + "\nH=\n" + str(H) + "\ndet(H)=" + str(
                                H.getDet()))
            # self.assertTrue(H == H_prime, "H=\n" + str(H) + "\ndet(H)=" + str(H.getDet()) + "\nH_prime=\n" + str(H_prime) + "\nA=\n" + str(A) +"\ndet(A)=" + str(A.getDet()) + "\nU=\n" + str(U)+ "\ndet(U)= " + str(U.getDet()))

    def test_hnf_fast(self):
        import numpy as np
        A = MyMatrix.getIdentity(10)
        H = hnf_fast(A)
        self.assertTrue(H.isHNF() and abs(H.getDet()) == abs(A.getDet()),
                        "A=\n" + str(A) + "\ndet(A)=" + str(A.getDet()) + "\nH=\n" + str(H) + "\ndet(H)=" + str(
                            H.getDet()))
        for i in range(TestMethods.max):
            # print(i)
            A = MyMatrix(np.random.randint(low=TestMethods.a, high=TestMethods.b, size=(TestMethods.n, TestMethods.n)).tolist())
            c = 0
            for j in range(1, TestMethods.n + 1):
                if A.getPrincipalSubmatrix(j).getDet() == 0:
                    c = 1

            if c == 1:
                continue
            import time
            time1 = time.time_ns()
            H = hnf_fast(A)
            time2 = time.time_ns()
            p = int(i / TestMethods.max * 100)
            s = "[" + "#" + "#" * (int(p/100 * 30)) + " " * (int((1 - p/100) * 30)) + "] " + str(p) + "%"
            print("\r" + '\033[92m' + s, end="", flush=True)
            print(" - " + '\033[0m' + "Execution time: ", (time2 - time1) / (10 ** 9), " seconds", end="", flush=True)
            # U = A.getInverse() @ H    # int too large to convert to float
            # U.ensureIntegers()
            # H_prime = A @ U
            # H_prime.ensureIntegers()
            self.assertTrue(H.isHNF() and abs(H.getDet()) == abs(A.getDet()),
                            "A=\n" + str(A) + "\ndet(A)=" + str(A.getDet()) + "\nH=\n" + str(H) + "\ndet(H)=" + str(
                                H.getDet()))
