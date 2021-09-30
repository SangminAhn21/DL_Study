from util_create_contexts_target import create_contexts_target
import sys
sys.path.append('/workspace/DL_Study/Deep_Learning_from_Scratch_2/DLfromScratch_2')
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=1)

print(contexts)

print(target)
