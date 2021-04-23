from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import os
import re
import torch

if not os.path.exists('kleister-nda-clone'):
    print('run\ngit clone https://github.com/applicaai/kleister-nda')
    exit()

NORMALISER = re.compile('(\\\\.|/s/)+')
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
INPUT = 'kleister-nda-clone/train/in.tsv'
TAGS = 'kleister-nda-clone/train/out.tsv'
JURISDICTIONS = {'Maine', 'Massachusetts', 'Ohio', 'Texas', 'Colorado', 'Oregon', 'New Jersey', 'South Carolina',
                 'Kansas', 'North_Carolina', 'Georgia', 'Minnesota', 'Florida', 'South Dakota', 'California',
                 'Virginia', 'Washington', 'Rhode Island', 'New York', 'Utah', 'Indiana', 'Idaho', 'Iowa',
                 'Pennsylvania', 'Connecticut', 'Delaware', 'Wisconsin', 'Michigan', 'Illinois', 'Nevada', 'Missouri'}
JURI_RE = re.compile('|'.join(JURISDICTIONS), flags=re.IGNORECASE)
months = '(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|June?|July?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)'
DATE1 = '([0-9][0-9])-([0-9][0-9])-([0-9][0-9][0-9][0-9])'
DATE2 = '([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])'
DATE3 = '([0-9][0-9]?)(st|rd|nd|th)?( day of |[ \'.,]{0,2})' + months + '[,\' .]{0,2}([0-9][0-9]([0-9][0-9])?)'  # 11
DATE4 = months + '[ \'.,]{0,2}([0-9][0-9]?)[,\' .]{1,2}([0-9][0-9]([0-9][0-9])?)'
DATE_RE = re.compile('|'.join([DATE1, DATE2, DATE3, DATE4]), flags=re.IGNORECASE)
if not os.path.isfile(INPUT):
    print('go to kleister-nda-clone/train/ and run\nxz --decompress in.tsv.xz')
    exit()


def jurisdiction_norm(s):
    return '_'.join([x[0].upper() + x[1:].lower() for x in s.split(' ')])


def party_norm(s):
    s = s.replace(',', '')
    s = '_'.join([x[0].upper() + x[1:].lower() for x in s.split(' ')])
    if s.endswith("Inc"):
        s = s + '.'
    s = s.replace('&','and')
    return s


RE1 = re.compile(DATE1, flags=re.IGNORECASE)
RE2 = re.compile(DATE2, flags=re.IGNORECASE)
RE3 = re.compile(DATE3, flags=re.IGNORECASE)
RE4 = re.compile(DATE4, flags=re.IGNORECASE)


def date_norm(s: str):
    m = RE1.match(s)
    if m is not None:
        return m.group(3) + '-' + m.group(1) + '-' + m.group(2)
    m = RE2.match(s)
    if m is not None:
        return s

    def get_m():
        mms = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, mm in enumerate(mms):
            if mm.lower() in s.lower():
                return ('0' if i < 9 else '') + str(i + 1)

    m = RE3.match(s)
    if m is not None:
        return m.group(14) + '-' + get_m() + '-' + m.group(1)

    m = RE4.match(s)
    if m is not None:
        return m.group(12) + '-' + get_m() + '-' + m.group(11)


def norm(s):
    return NORMALISER.sub(' ', s)


def train():
    with open(INPUT) as in_f, open('kleister-nda-clone/train/expected.tsv') as ex_f:
        correct_juris = 0
        incorrect_juris = 0
        correct_date = 0
        incorrect_date = 0
        correct_party = 0
        incorrect_party = 0
        for in_line, ex_line in zip(in_f, ex_f):
            filename, keys, text, _, _, _ = in_line.split('\t')
            ex_line = ex_line.strip()
            norm_text = norm(text)
            jurisdiction = None
            parties = []
            date = None
            for query in ex_line.split(' '):
                key, value = query.split('=')
                if key == 'effective_date':
                    date = value
                elif key == 'party':
                    parties.append(value)
                elif key == 'jurisdiction':
                    jurisdiction = value
            if jurisdiction is not None:
                found_juri = JURI_RE.search(norm_text)
                found_juri = None if found_juri is None else found_juri.group()

                if found_juri is not None and jurisdiction_norm(found_juri) == jurisdiction:
                    correct_juris += 1
                else:
                    incorrect_juris += 1
                print(jurisdiction, '==', found_juri)
            if date is not None:
                found_date = DATE_RE.search(norm_text)
                found_date_ = None if found_date is None else found_date.group()
                found_date = None if found_date_ is None else date_norm(found_date_)
                if date == found_date:
                    correct_date += 1
                else:
                    incorrect_date += 1
                print(date, '==', found_date, found_date_)
            if len(parties) > 0:
                found_parties = bert(norm_text)
                wrong = 0
                for expected_party in parties:
                    if expected_party in found_parties:
                        correct_party += 1
                    else:
                        incorrect_party += 1
                        wrong += 1
                print(parties, ' == ', found_parties, '===', wrong*'%')
        print('juri correct=', correct_juris, 'incorrect=', incorrect_juris, '%=',
              correct_juris / (correct_juris + incorrect_juris))
        print('date correct=', correct_date, 'incorrect=', incorrect_date, '%=',
              correct_date / (correct_date + incorrect_date))
        print('party correct=', correct_party, 'incorrect=', incorrect_party, '%=',
              correct_party / (correct_party + incorrect_party))


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)


def bert(norm_text):
    prev_start = 0
    prev_end = 0
    prev_entity = ''
    compound = ''
    all_tags = []
    tags = nlp(norm_text)

    for tag in tags:
        entity = tag['entity']
        entity = entity[2:]  # remove B- and I- prefixes
        entity = 'ORG' if entity == 'PER' else entity  # We don't need to differentiate between PER and ORG
        start = tag['start']
        end = tag['end']
        string = norm_text[start:end]
        if prev_entity == entity and (prev_end == start or prev_end + 1 == start):
            if prev_end == start:
                compound += string
            elif prev_end + 1 == start:
                compound += ' ' + string
        else:
            all_tags.append((compound, prev_entity, prev_start, prev_end))
            prev_start = start
            compound = string
        prev_entity = entity
        prev_end = end

    all_tags.append((compound, prev_entity, prev_start, prev_end))
    all_tags = [x[0] for x in all_tags if x[1] == "ORG"
                and "Company" not in x[0]
                and "Receiving" not in x[0]
                and "Directors" not in x[0]
                and "Party" not in x[0]]
    all_tags.sort(key=lambda x: len(x), reverse=True)
    all_tags = list(map(party_norm, all_tags))
    unique = set()
    dedup = []
    for tag in all_tags:
        l = tag.lower()
        if l not in unique:
            unique.add(l)
            dedup.append(tag)
    all_tags = dedup[:2]
    all_tags.sort()
    return all_tags


train()

exit()
