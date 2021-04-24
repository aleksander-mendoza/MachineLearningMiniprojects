from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import os
import re
import torch
from itertools import cycle

if not os.path.exists('kleister-nda-clone'):
    print('run\ngit clone https://github.com/applicaai/kleister-nda')
    exit()

NORMALISER = re.compile('(\\\\.|/s/)+')
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
MODE = 'train'  # either 'test-A' or 'train' or 'dev-0'
INPUT = 'kleister-nda-clone/' + MODE + '/in.tsv'
OUTPUT = 'kleister-nda-clone/' + MODE + '/out.tsv'
EXPECTED = 'kleister-nda-clone/' + MODE + '/expected.tsv'
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

TERM_1 = '\\([0-9]{1,3}\\) (days?|years?|months?)'
TERM_2 = '((a|one|two|three|four|five|(six|seven|eight|nine)(teen|ty ?)?|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty ?|thirty ?|forty ?|fifty ?)+) (days?|years?|months?)'
TERM_RE = re.compile('|'.join([TERM_1, TERM_2]))
TERM_RE1 = re.compile(TERM_1)
TERM_RE2 = re.compile(TERM_2)
ATTENTION_TO_KEYWORD = re.compile('period of', flags=re.IGNORECASE)
LETTER_RE = re.compile('[a-zA-Z]+')
NUMBER_CONTINUATION_RE = re.compile('#(?=[0-9])')

if not os.path.isfile(INPUT):
    print('go to kleister-nda-clone/' + MODE + '/ and run\nxz --decompress in.tsv.xz')
    exit()


def term_norm(s):
    m = TERM_RE2.match(s)
    if m is not None:
        w = m.group(1)
        w = w.replace('ten','10').replace('eleven','11').replace('twelve','12').replace('thirteen','13')\
            .replace('fourteen','14').replace('fifteen','15').replace('sixteen','16').replace('seventeen','17')\
            .replace('eighteen','18').replace('nineteen','19').replace('twenty','2#').replace('thirty','3#')\
            .replace('forty','4#').replace('fifty','5#').replace('sixty','6#').replace('seventy','7#')\
            .replace('eighty','8#').replace('ninety','9#').replace('one','1')\
            .replace('two', '2').replace('three','4').replace('four','4').replace('five','5')\
            .replace('six','6').replace('seven','7').replace('eight','8').replace('nine','9')
        w = LETTER_RE.sub('', w)
        w = NUMBER_CONTINUATION_RE.sub('', w)
        w = w.replace('#','0')
        w = w+'_'+m.group(5)
        return w
    return s.lower().replace(' ', '_').replace('(', '').replace(')', '')


def jurisdiction_norm(s):
    return '_'.join([x[0].upper() + x[1:].lower() for x in s.split(' ')])


def party_norm(s:str):
    def upper_first(w):
        assert len(w)>0, s
        if len(w) == 1:
            return w.upper()
        return w[0].upper() + w[1:].lower()

    s = s.replace(',', '')
    s = '_'.join([upper_first(x) for x in s.strip().split()])
    if s.endswith("Inc"):
        s = s + '.'
    s = s.replace('&', 'and')
    return s


RE1 = re.compile(DATE1, flags=re.IGNORECASE)
RE2 = re.compile(DATE2, flags=re.IGNORECASE)
RE3 = re.compile(DATE3, flags=re.IGNORECASE)
RE4 = re.compile(DATE4, flags=re.IGNORECASE)


def date_norm(s: str):
    def pad_zero(w):
        return ('0' if int(w) < 10 else '') + str(w)

    m = RE1.match(s)
    if m is not None:
        return m.group(3) + '-' + pad_zero(m.group(1)) + '-' + pad_zero(m.group(2))
    m = RE2.match(s)
    if m is not None:
        return s

    def get_m():
        mms = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, mm in enumerate(mms):
            if mm.lower() in s.lower():
                return pad_zero(i + 1)

    m = RE3.match(s)
    if m is not None:
        return m.group(14) + '-' + get_m() + '-' + pad_zero(m.group(1))

    m = RE4.match(s)
    if m is not None:
        return m.group(12) + '-' + get_m() + '-' + pad_zero(m.group(11))


def norm(s):
    return NORMALISER.sub(' ', s)


def train():
    if os.path.isfile(EXPECTED):
        ex_f = open(EXPECTED)
    else:
        ex_f = None
    with open(INPUT) as in_f, open(OUTPUT, 'w+') as out_f:
        correct_juris = 0
        incorrect_juris = 0
        correct_date = 0
        incorrect_date = 0
        correct_party = 0
        incorrect_party = 0
        correct_terms = 0
        incorrect_terms = 0
        for in_line, ex_line in zip(in_f, ex_f if ex_f is not None else cycle([None])):
            filename, keys, text, _, _, _ = in_line.split('\t')
            norm_text = norm(text)
            jurisdiction = None
            parties = []
            date = None
            term = None
            output = []
            for query in ex_line.strip().split(' ') if ex_line else keys.split(' '):
                key, value = query.split('=') if ex_line else (query, True)
                if key == 'effective_date':
                    date = value
                elif key == 'party':
                    parties.append(value)
                elif key == 'jurisdiction':
                    jurisdiction = value
                elif key == 'term':
                    term = value

            if jurisdiction is not None:
                found_juri = JURI_RE.search(norm_text)
                found_juri = None if found_juri is None else jurisdiction_norm(found_juri.group())
                if ex_f:
                    if found_juri == jurisdiction:
                        correct_juris += 1
                    else:
                        incorrect_juris += 1
                        print(jurisdiction, '!=', found_juri)
                if found_juri is not None:
                    output.append('jurisdiction=' + found_juri)
            if term is not None:
                attention = ATTENTION_TO_KEYWORD.match(norm_text)
                if attention:
                    found_term = TERM_RE.search(norm_text[attention.end():attention.end()+30])
                else:
                    found_term = TERM_RE.search(norm_text)

                found_term = None if found_term is None else term_norm(found_term.group())
                if ex_f:
                    if found_term == term:
                        correct_terms += 1
                    else:
                        incorrect_terms += 1
                        print('TERM', term, '!=', found_term)
                if found_term is not None:
                    output.append('term=' + found_term)
            if date is not None:
                found_date = DATE_RE.search(norm_text)
                found_date = None if found_date is None else date_norm(found_date.group())
                if ex_f:
                    if date == found_date:
                        correct_date += 1
                    else:
                        incorrect_date += 1
                        print('DATE', date, '!=', found_date)
                if found_date is not None:
                    output.append('effective_date=' + found_date)
            if len(parties) > 0:
                found_parties = bert(norm_text)
                wrong = 0
                if ex_f:
                    for expected_party in parties:
                        if expected_party in found_parties:
                            correct_party += 1
                        else:
                            incorrect_party += 1
                            wrong += 1
                            print('PARTY', expected_party, 'not in', found_parties)
                output += ['party=' + party for party in found_parties]
            out_f.write(' '.join(output) + '\n')
        if ex_f:
            print('term correct=', correct_terms, 'incorrect=', incorrect_terms, '%=',
                  correct_terms / (correct_terms + incorrect_terms))
            print('juri correct=', correct_juris, 'incorrect=', incorrect_juris, '%=',
                  correct_juris / (correct_juris + incorrect_juris))
            print('date correct=', correct_date, 'incorrect=', incorrect_date, '%=',
                  correct_date / (correct_date + incorrect_date))
            print('party correct=', correct_party, 'incorrect=', incorrect_party, '%=',
                  correct_party / (correct_party + incorrect_party))
            ex_f.close()



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
