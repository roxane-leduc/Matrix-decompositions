import numpy as np
import scipy.linalg as sp

TRAIT = f'\n{"x"*80}'
float_formatter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def init(n):
	A = np.ones([n, n])
	for j in range(n):
		A[j,j] = n
	A = sp.hilbert(n)
	b = np.ones(n)
	return A, b

def _lu(A, b, n):
	M, invM = {}, {}
	I = np.diag(np.diag(np.ones([n, n])))
	for i in range(0, n-1):
		M[i] = np.zeros([n,n])
		for j in range(n):
			M[i][j,j] = 1
		for k in range(i+1, n):
			M[i][k,i] = -A[k,i]/A[i,i]
		invM[i] = 2*I - M[i]
		A = np.matmul(M[i], A)
		print(f'M[k={i}] = \n{M[i]}\n')
		print(f'M[{i}] * ... * A = \n{A}\n{TRAIT}')
	return M, A, invM

#~ =====
#~ MAIN
n = 4
A,b = init(n)
AA = A.copy()
print(f'Resol Ax=b avec:\nA = {A}\nb = {b}\n{TRAIT}')

M, A, invM = _lu(A, b, n)

MM = M[0]
for i in range(1, n-1):
	MM = np.matmul(M[i],MM)

L = invM[0]
for i in range(1, n-1):
	L = np.matmul(L, invM[i])

print(f'MM =\n {MM}\n')
print(f'U = MM * A =\n {A}\n')
print(f'L =\n {L}\n')

print(f'A - LU = {AA - np.matmul(L, A)}')
#~ print(np.matmul(MM, A))
