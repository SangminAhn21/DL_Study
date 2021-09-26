import sys
sys.path.append('/workspace/DL_Study/Deep_Learning_from_Scratch_2/DLfromScratch_2')
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape)
print('t', t.shape)