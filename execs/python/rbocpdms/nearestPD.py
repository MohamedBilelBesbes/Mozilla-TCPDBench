from numpy import linalg as la
import numpy as np

class NPD:
    def isPD(B):
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    def nearestPD(A):
        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if NPD.isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))

        I = np.eye(A.shape[0])
        k = 1
        while not NPD.isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3
