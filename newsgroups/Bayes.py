from tqdm import tqdm
import re
import math
from math import log, exp
from sklearn.datasets import fetch_20newsgroups
import gensim

newsgroups = fetch_20newsgroups()
newsgroups_text = newsgroups['data']
Y = newsgroups['target']
Y_names = newsgroups['target_names']

max_class = 19
no_classes = max_class + 1
lexicon = {}
number_of_docs_per_class = [0] * no_classes

obfuscator = re.compile('[\\[?.,!()\\]*&^%$#@{}|\\\\/~\\- \t\n]+')


def tokenize(txt):
    return list(set(gensim.utils.tokenize(txt, lowercase=True)))


def train():
    for line, clazz in tqdm(zip(newsgroups_text, Y), desc="training"):
        clazz = int(clazz)
        for lemma in tokenize(line):
            classes = lexicon.get(lemma)
            if not classes:
                classes = [0] * no_classes
                lexicon[lemma] = classes
            classes[clazz] += 1
        number_of_docs_per_class[clazz] += 1


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def classify(doc):
    number_of_docs = sum(number_of_docs_per_class)
    p_of_class_a_priori = [num / number_of_docs for num in number_of_docs_per_class]
    log_p_words_and_class = [log(class_prob) for class_prob in p_of_class_a_priori]
    for word in doc:
        class_docs_with_word = lexicon.get(word)
        if class_docs_with_word:
            for clazz, word_freq in enumerate(class_docs_with_word):
                log_p_word_given_class = log(word_freq + 1) - log(number_of_docs_per_class[clazz] + no_classes)
                log_p_words_and_class[clazz] += log_p_word_given_class
    p_words = sum(map(exp, log_p_words_and_class))
    probability_of_class = [exp(log_prob) / p_words for log_prob in log_p_words_and_class]
    return argmax(log_p_words_and_class), probability_of_class


def get_prob(index, words):
    _, probs = classify(words)
    return probs[index]


def print_probs(probs):
    for prob, name in sorted(zip(probs, Y_names), key=lambda x: x[0], reverse=True):
        print("%.5f" % prob, '\t\t', name)

    print("%.5f" % sum(probs), '\t\ttotal', )


train()
print_probs(classify({'i', 'love', 'guns'})[1])

# 0.01435 		 alt.atheism
# 0.00155 		 comp.graphics
# 0.00268 		 comp.os.ms-windows.misc
# 0.00578 		 comp.sys.ibm.pc.hardware
# 0.00081 		 comp.sys.mac.hardware
# 0.00110 		 comp.windows.x
# 0.00417 		 misc.forsale
# 0.00799 		 rec.autos
# 0.02284 		 rec.motorcycles
# 0.00220 		 rec.sport.baseball
# 0.00677 		 rec.sport.hockey
# 0.00761 		 sci.crypt
# 0.00333 		 sci.electronics
# 0.00171 		 sci.med
# 0.00511 		 sci.space
# 0.01015 		 soc.religion.christian
# 0.76589 		 talk.politics.guns
# 0.03066 		 talk.politics.mideast
# 0.05010 		 talk.politics.misc
# 0.05520 		 talk.religion.misc
# 1.00000 		total

print_probs(classify({'is', 'there', 'life', 'after', 'death'})[1])

# 0.09929 		 alt.atheism
# 0.00029 		 comp.graphics
# 0.00065 		 comp.os.ms-windows.misc
# 0.00066 		 comp.sys.ibm.pc.hardware
# 0.00178 		 comp.sys.mac.hardware
# 0.00070 		 comp.windows.x
# 0.00005 		 misc.forsale
# 0.00662 		 rec.autos
# 0.00428 		 rec.motorcycles
# 0.00087 		 rec.sport.baseball
# 0.00096 		 rec.sport.hockey
# 0.00263 		 sci.crypt
# 0.00035 		 sci.electronics
# 0.01348 		 sci.med
# 0.01344 		 sci.space
# 0.30850 		 soc.religion.christian
# 0.10385 		 talk.politics.guns
# 0.17518 		 talk.politics.mideast
# 0.08120 		 talk.politics.misc
# 0.18523 		 talk.religion.misc
# 1.00000 		total