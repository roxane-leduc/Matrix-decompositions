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

def _build_householder_matrix(u, n):
	I = np.diag(np.diag(np.ones([n, n])))
	return I - 2 * np.matmul(u.reshape(n,1), u.reshape(1,n)) / (np.linalg.norm(u)**2)

	#~ ATTENTION CE QUI SUIT NE FONCTIONNE PAS !
	#~ return I - 2 * np.matmul(u, np.transpose(u)) / (np.linalg.norm(u)**2)

def _householder(A, n):
	#~ https://fr.wikipedia.org/wiki/D%C3%A9composition_QR#M%C3%A9thode_de_Householder
	H = [None] * (n-1)
	R = A.copy()

	for c in range(n-1):
		
		u = R[:, c]
		alpha = np.sign(u[0])
		for i in range(c):
			u[i] = 0
		u[c] = u[c] - alpha * np.linalg.norm(u)

		H[c] = _build_householder_matrix(u, n)
		R = np.matmul(H[c], R)

	Q = H[0]
	for c in range(1, n-1):
		Q = np.matmul(Q, H[c])

	#~ LA LIGNE QUI SUIT EST UNE CORRECTION IMPERATIVE !
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
n = 6
A = init(n)
_print('A')
print(TRAIT)

Q, R = _householder(A, n)

print(TRAIT)
_print('Q')
_print('R')

print(TRAIT)
_print("A - Q*R", A - np.matmul(Q, R))
