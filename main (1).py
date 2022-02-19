import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import math
import random

def generateRandomBPSKvector(n):
  symbols = [-1,1]
  vector = []
  for i in range(n):
    vector.append(random.choice(symbols))
  return np.array(vector)

def generateRandomQAM4vector(n):
  symbols = (2**0.5)/2 * np.array([+1+1j, +1-1j, -1+1j, -1-1j])
  vector = []
  for i in range(n):
    vector.append(random.choice(symbols))
  return np.array(vector)

def generateWhiteNoiseVector(n):
  mean = 0
  std = 1 
  num_samples = n
  vector = np.random.normal(mean, std, size=num_samples)
  return vector 

L = 41
M = 32
P = M + L - 1

def generateH():
  h = []
  for i in range(L):
    h.append(random.random())
  h = np.array(h)
  norm = np.linalg.norm(h)
  h = h/norm
  padding = np.zeros(h.shape[0] - 1, h.dtype)
  first_col = np.r_[h, padding]
  first_row = np.r_[h[0], padding]
  H = toeplitz(first_col, first_row)
  H = H[0:P,0:M]
  return np.array(H)

def BPSKEstimator(M):
  vector = []
  for i in M:
    if i > 0:
      vector.append(1)
    else:
      vector.append(-1)
  return np.array(vector)

def QAM4Estimator(M):
  vector = []
  for i in M:
    angle_i = np.angle(i)
    if angle_i > 0 and angle_i < math.pi/2:
      vector.append(1+1j)
    elif angle_i > math.pi/2 and angle_i < math.pi:
      vector.append(-1+1j)
    elif angle_i < 0 and angle_i > -math.pi/2:
      vector.append(1-1j)
    else:
      vector.append(-1-1j)
  return np.array(vector)*(2**(1/2))/2

def calculateErrorProbability(tx, rx):
  errors = 0
  for i in range(len(tx)):
    if tx[i] != rx[i]:
      errors += 1
  return errors/len(tx)

def calculateSNR(signal, noise):
  return np.var(signal)/np.var(noise)

y = np.empty(10000)
x = np.empty(10000)

for i in range(0,10000):
  H   = generateH()
  un  = generateRandomBPSKvector(M)
  #un  = generateRandomQAM4vector(M)
  #un  = np.transpose(un)
  v0n = generateWhiteNoiseVector(P)
  zn  = np.dot(H,un)
  y0n = zn + v0n
  K   = np.linalg.pinv(H)
  eSn = np.dot(K,y0n)
  iSn = BPSKEstimator(eSn)
  #iSn = QAM4Estimator(eSn)
  errorProbability = calculateErrorProbability(un,iSn)
  SNR              = calculateSNR(zn,v0n)
  y[i] = errorProbability
  x[i] = SNR

plt.plot(x,y,'o')
plt.savefig('resultBPSK.png')

plt.clf()

for i in range(0,10000):
  H   = generateH()
  #un  = generateRandomBPSKvector(M)
  un  = generateRandomQAM4vector(M)
  #un  = np.transpose(un)
  v0n = generateWhiteNoiseVector(P)
  zn  = np.dot(H,un)
  y0n = zn + v0n
  K   = np.linalg.pinv(H)
  eSn = np.dot(K,y0n)
  #iSn = BPSKEstimator(eSn)
  iSn = QAM4Estimator(eSn)
  errorProbability = calculateErrorProbability(un,iSn)
  SNR              = calculateSNR(zn,v0n)
  y[i] = errorProbability
  x[i] = SNR

plt.plot(x,y,'o')
plt.savefig('resultQAM4.png')