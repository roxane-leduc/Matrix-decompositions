import numpy as np
import scipy.linalg as sp

TRAIT = f'\n{"x"*80}'

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def _print(Mlabel, M=None):
	if M is None:
		M = eval(Mlabel)
	print(f'{Mlabel} =\n{M}\n')

def init(n):
	A = sp.hilbert(n)
	return A

def _proj(e, a):
	return np.dot(e, a) * e

def _gram_schmidt(A, n):
	#~ https://fr.wikipedia.org/wiki/D%C3%A9composition_QR
	E = np.zeros([n, n]) * np.nan
	U = np.zeros([n, n]) * np.nan

	for c in range(n):

		U[:,c] = A[:,c]
		for k in range(c):
			U[:,c] = U[:,c] - _proj(E[:,k], A[:,c])

		E[:,c] = U[:,c] / np.linalg.norm(U[:,c])
			
		_print('E', E)
		_print('U', U)

	Q = E
	R = np.matmul(np.transpose(Q), A)
		
	return Q, R
#~ =====
#~ MAIN
n = 3
A = np.array([
[12, -51, 4],
[6, 167, -68],
[-4, 24, -41],
])
#~ A = init(n)
_print('A')

Q, R = _gram_schmidt(A, n)
_print('Q')
_print('R')

_print("A - Q*R", A - np.matmul(Q, R))