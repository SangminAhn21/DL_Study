import numpy as np
import sys
sys.path.append('/workspace/DL_Study/Deep_Learning_from_Scratch_2/DLfromScratch_2')
from ch04.negative_sampling_layer import UnigramSampler

words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
print(np.random.choice(words))
print(np.random.choice(words, size=5))  # 중복 있음
print(np.random.choice(words, size=5, replace=False))  # 중복 없음
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
print(np.random.choice(words, p=p))

corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus, power, sample_size)
target = np.array([1, 3, 0])
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)
