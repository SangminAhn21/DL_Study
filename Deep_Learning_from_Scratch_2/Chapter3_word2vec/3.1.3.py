import sys
sys.path.append('/workspace/DL_Study/Deep_Learning_from_Scratch_2/DLfromScratch_2')
from common.layers import MatMul
import numpy as np

C = np.array([[1, 0, 0, 0, 0, 0, 0]])
print(C.shape)
W = np.random.randn(7, 3)
layer = MatMul(W)
h = layer.forward(C)
print(h)
