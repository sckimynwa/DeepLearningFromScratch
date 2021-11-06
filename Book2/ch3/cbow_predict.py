import sys
sys.path.append('..')
from common.util import preprocess, create_contexts_target, convert_one_hot

# Preprecess Text
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print('===== Word Preprocess =====')
print(corpus)
print(word_to_id)
print(id_to_word)
print('\n')

print('===== Context and Targets =====')
contexts, target  = create_contexts_target(corpus, window_size=1)
print(contexts)
print(target)
print('\n')

print('===== Convert to One-Hot Vector =====')
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size=vocab_size)
contexts = convert_one_hot(contexts, vocab_size=vocab_size)
print(contexts)
print(target)

print('test')
print(contexts.ndim)
print(contexts[:, 0].ndim)