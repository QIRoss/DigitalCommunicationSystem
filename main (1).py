import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import math
import random
import pandas as pd

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

def generateWhiteNoiseVector(n, std=1):
  mean = 0
  std = random.uniform(0.14, 1)
  num_samples = n
  vector = np.random.normal(mean, std, size=num_samples)
  return vector 

L = 41
M = 8
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
    if angle_i >= 0 and angle_i <= math.pi/2:
      vector.append(1+1j)
    elif angle_i >= math.pi/2 and angle_i <= math.pi:
      vector.append(-1+1j)
    elif angle_i <= 0 and angle_i >= -math.pi/2:
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

aaa = np.empty(shape=(10000,2))

for i in range(0,10000):
  H   = generateH()
  un  = generateRandomBPSKvector(M)
  v0n = generateWhiteNoiseVector(P)
  zn  = np.dot(H,un)
  y0n = zn + v0n
  K   = np.linalg.pinv(H)
  eSn = np.dot(K,y0n)
  iSn = BPSKEstimator(eSn)
  errorProbability = calculateErrorProbability(un,iSn)
  SNR              = calculateSNR(zn,v0n)
  aaa[i][0] = np.round(SNR,1)
  aaa[i][1] = errorProbability

aaa1 = pd.DataFrame(aaa).groupby(0, as_index=False)[1].mean().values.tolist()
aaa2 = [i[1] for i in aaa1]
aaa3 = [i[0] for i in aaa1]

plt.plot(aaa3,aaa2, color='blue')

for i in range(0,10000):
  H   = generateH()
  un  = generateRandomQAM4vector(M)
  v0n = generateWhiteNoiseVector(P)
  zn  = np.dot(H,un)
  y0n = zn + v0n
  K   = np.linalg.pinv(H)
  eSn = np.dot(K,y0n)
  iSn = QAM4Estimator(eSn)
  errorProbability = calculateErrorProbability(un,iSn)
  SNR              = calculateSNR(zn,v0n)
  aaa[i][0] = np.round(SNR,1)
  aaa[i][1] = errorProbability

aaa1 = pd.DataFrame(aaa).groupby(0, as_index=False)[1].mean().values.tolist()
aaa2 = [i[1] for i in aaa1]
aaa3 = [i[0] for i in aaa1]

plt.plot(aaa3,aaa2,color='red')
plt.savefig('Result.png')
