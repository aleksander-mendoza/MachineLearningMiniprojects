from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import os
import re
import torch

if not os.path.exists('kleister-nda-clone'):
    print('run\ngit clone https://github.com/applicaai/kleister-nda')
    exit()

INPUT = 'kleister-nda-clone/train/in.tsv'
TAGS = 'kleister-nda-clone/train/tags.tsv'

if not os.path.isfile(INPUT):
    print('go to kleister-nda-clone/train/ and run\nxz --decompress in.tsv.xz')
    exit()

NORMALISER = re.compile('(\\\\.|/s/|[\\W_0-9])+')


def norm(s):
    return NORMALISER.sub(' ', s)


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)

with open(INPUT) as in_f, open('kleister-nda-clone/train/expected.tsv') as ex_f:
    for in_line, ex_line in zip(in_f, ex_f):
        filename, keys, text, _, _, _ = in_line.split('\t')
        ex_line = ex_line.strip()
        text = norm(text)
        print("=====", ex_line, "=====")
        prev_start = 0
        prev_end = 0
        prev_entity = ''
        compound = ''
        for tag in nlp(text):
            entity = tag['entity']
            entity = entity[2:]  # remove B- and I- prefixes
            entity = 'ORG' if entity == 'PER' else entity  # We don't need to differentiate between PER and ORG
            start = tag['start']
            end = tag['end']
            string = text[start:end]
            if prev_entity == entity and (prev_end == start or prev_end + 1 == start):
                if prev_end == start:
                    compound += string
                elif prev_end + 1 == start:
                    compound += ' ' + string
            else:
                print(compound, prev_entity, prev_start, prev_end)
                prev_start = start
                compound = string
            prev_entity = entity
            prev_end = end
