import numpy as np
import scipy.linalg as sp

TRAIT = f'\n{"x"*80}'

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def _print(Mlabel, M=None):
	if M is None:
		M = eval(Mlabel)
	print(f'{Mlabel} =\n{M}\n')

def init(n):
	A = sp.hilbert(n)
	return A

def is_pos_def(A):
	return np.all(np.linalg.eigvals(A) > 0)

def _choleski(A, n):
	L = np.zeros([n, n]) # * np.nan

	for k in range(n):
		L[k,k] = A[k,k]
		for i in range(k):
			L[k,k] = L[k,k] - (L[k,i] ** 2)
		L[k,k] = np.sqrt(L[k,k])

		for i in range(k+1, n):
			L[i,k] = A[i,k]
			for j in range(k):
				L[i,k] = L[i,k] - L[i,j] * L[k,j]
			L[i,k] = L[i,k] / L[k,k]

		_print(f'k={k}, L', L)
	return L

#~ =====
#~ MAIN
n = 4
A = init(n)
A = np.array([
[1, 1, 1, 1],
[1, 5, 5, 5],
[1, 5, 14, 14],
[1, 5, 14, 15]
])
_print('A')

if is_pos_def(A):
	L = _choleski(A, n)
	#~ _print('L')

	Lt = np.transpose(L)
	#~ _print("L'", Lt)

	_print("A - L*L'", A - np.matmul(L, Lt))
else:
	print('A non SDP !')
