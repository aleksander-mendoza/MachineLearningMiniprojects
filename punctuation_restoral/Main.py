# With help from tutorial
# https://huggingface.co/transformers/training.html
from transformers import BertForMaskedLM, BertTokenizer, pipeline, BertForTokenClassification, BertTokenizerFast
import re
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
import os
import sys

##################################################################################
# Get data from https://github.com/poleval/2021-punctuation-restoration
##################################################################################

import re

fixer = re.compile("([^ ]) +([.,?!:;-]+)(.)")


# def fix_bert_spaces(lo):
#     start = 0
#     new_lo = ""
#     for m in fixer.finditer(lo):
#         before = m.group(1)
#         punc = m.group(2)
#         after = m.group(3)
#         end, newstart = m.span()
#         new_lo += lo[start:end]
#         if before in ['.', '?', '!', ':', ';', ',', '-']:
#             new_lo += before
#         else:
#             new_lo += before + punc
#         if after.isspace():
#             new_lo += after
#         else:
#             new_lo += " " + after
#         start = newstart
#     new_lo += lo[start:]
#     return new_lo


# with open('2021-punctuation-restoration/train/out.tsv') as o, \
#         open('2021-punctuation-restoration/train/expected.tsv') as e, \
#         open('2021-punctuation-restoration/train/in.tsv') as i:
#     for lo, le, li in zip(o, e, i):
#         lo = fix_bert_spaces(lo)
#         lo = lo.split(' ')
#         le = le.split(' ')
#
#         if len(lo) != len(le):
#             out_o = []
#             out_i = []
#             out_e = []
#             for i in range(min(len(lo), len(le))):
#                 if lo[i].rstrip(".?,!:;-") != le[i].rstrip(".?,!:;-"):
#                     out_o.append(lo[i])
#                     out_e.append(le[i])
#
#             print("o=", out_o)
#             print("e=", out_e)
#             print("lo=", lo)
#             print("le=", le)
#             print("i=", li)
# exit()
id_to_lbl = ['', '.', '?', '!', ':', ';', ',', '-', '...']
LABELS = {'.': 1, '?': 2, '!': 3, ':': 4, ';': 5, ',': 6, '-': 7, '...': 8, '': 0}
assert len(LABELS) == len(id_to_lbl)
for lbl, idx in LABELS.items():
    assert LABELS[lbl] == idx
sep = re.compile('[ .?!:;,"-]+')
DATA_WORDS = []
DATA_LABELS = []
TRAINING_MODE = False

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = BertTokenizerFast.from_pretrained("dkleczek/bert-base-polish-cased-v1")
trouble_tokens = []
for t in 'ニ尾の化け三大亀五イルカ馬八尾六尾の蛞蝓':
    if tokenizer.add_tokens([t]) == 1:
        trouble_tokens.append(tokenizer.vocab[t])
trouble_tokens = set(trouble_tokens)

if os.path.isdir('checkpoints') and os.listdir('checkpoints') != []:
    last_model = max([int(x[len('epoch_'):]) for x in os.listdir('checkpoints')])
    saved_model_path = './checkpoints/epoch_' + str(last_model)
else:
    saved_model_path = "dkleczek/bert-base-polish-cased-v1"
model = BertForTokenClassification.from_pretrained(saved_model_path, num_labels=len(LABELS))
model.to(DEVICE)

if TRAINING_MODE:
    model.train()

    with open('2021-punctuation-restoration/train/expected.tsv') as f:
        for line in tqdm(f, desc="Processing", total=800):
            line = line.strip()
            prev_idx = 0
            SENTENCE_WORDS = []
            SENTENCE_LABELS = []
            for match in sep.finditer(line):
                word = line[prev_idx:match.start()]
                if word == "":
                    continue
                punct = match.group().replace(' ', '')
                has_quote = False
                if '"' in punct:
                    punct = punct.replace('"', '')
                    has_quote = True
                if len(punct) > 1 and punct != '...':
                    punct = punct[0]
                label = LABELS[punct]
                # print(word + ':' + punct + '=' + str(label) + ('"' if has_quote else ''))
                prev_idx = match.end()
                SENTENCE_WORDS.append(word)
                SENTENCE_LABELS.append(label)
            DATA_WORDS.append(SENTENCE_WORDS)
            DATA_LABELS.append(SENTENCE_LABELS)

    train_texts, val_texts, train_tags, val_tags = train_test_split(DATA_WORDS, DATA_LABELS, test_size=.2)

    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                              truncation=True)


    def encode_tags(tags, encodings):
        encoded_labels = []
        for doc_labels, doc_offset, input_ids in zip(tags, encodings.offset_mapping, encodings.input_ids):
            # create an empty array of -100
            doc_enc_labels = torch.ones(len(doc_offset), dtype=torch.int) * -100
            arr_offset = torch.tensor(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = torch.tensor(doc_labels,
                                                                                             dtype=torch.int)
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels


    train_labels = encode_tags(train_tags, train_encodings)
    val_labels = encode_tags(val_tags, val_encodings)


    class PolEvalDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)


    train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
    val_encodings.pop("offset_mapping")

    train_dataset = PolEvalDataset(train_encodings, train_labels)
    val_dataset = PolEvalDataset(val_encodings, val_labels)

    FREEZE_BERT = False
    if FREEZE_BERT:
        PHYSICAL_BATCH_SIZE = 8
        VIRTUAL_BATCH_SIZE = 8
        for param in model.bert.parameters():
            param.requires_grad = False
    else:
        PHYSICAL_BATCH_SIZE = 1
        VIRTUAL_BATCH_SIZE = 8
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)


    def f_score(table):
        with torch.no_grad():
            true_positive = table.diag()
            total_predictions = table.sum(dim=1)  # = true positive + false positive
            total_labels = table.sum(dim=0)  # = true positive + false negative
            precision = true_positive / total_predictions
            recall = true_positive / total_labels
            f1_score = 2 * precision * recall / (precision + recall)
            return f1_score


    def add_to_prediction_table(table, logits, labels):
        for prediction, actual in zip(logits.argmax(dim=2).reshape(-1), labels.reshape(-1)):
            if actual != -100:
                table[prediction, actual] += 1


    def print_summary(epoch, train_table, val_table, fd):
        print("Training " + str(epoch), file=fd)
        print(train_table.int(), file=fd)
        print(list(zip(id_to_lbl, map(lambda x: x.item(), f_score(train_table)))), file=fd)
        print("Validation " + str(epoch), file=fd)
        print(val_table.int(), file=fd)
        print(list(zip(id_to_lbl, map(lambda x: x.item(), f_score(val_table)))), file=fd)
        fd.flush()


    with open('training.log', 'a+') as f:
        for epoch in range(100):
            train_prediction_table = torch.zeros(len(LABELS), len(LABELS))
            virtual_batch = 0
            optim.zero_grad()
            for batch in tqdm(train_loader, desc="training"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                add_to_prediction_table(train_prediction_table, outputs.logits, labels)
                loss.backward()
                virtual_batch += PHYSICAL_BATCH_SIZE
                if virtual_batch >= VIRTUAL_BATCH_SIZE:
                    optim.step()
                    optim.zero_grad()
                    virtual_batch = 0
            if virtual_batch > 0:
                optim.step()
                optim.zero_grad()
            model.save_pretrained('checkpoints/epoch_' + str(epoch))
            val_prediction_table = torch.zeros(len(LABELS), len(LABELS))
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="validation"):
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['labels']
                    outputs = model(input_ids, attention_mask=attention_mask)
                    add_to_prediction_table(val_prediction_table, outputs.logits, labels)
            print_summary(epoch, train_prediction_table, val_prediction_table, f)
            print_summary(epoch, train_prediction_table, val_prediction_table, sys.stdout)
else:  # TRAINING_MODE == False
    model.eval()
    DIR = 'test-A'
    with open('2021-punctuation-restoration/' + DIR + '/in.tsv') as f, open(
            '2021-punctuation-restoration/' + DIR + '/out.tsv', 'w+') as o:
        for line in tqdm(f, desc="Processing", total=200):
            _, line = line.strip().split('\t')
            encoded = tokenizer([line], padding=True, truncation=True)
            input_ids = torch.tensor(encoded['input_ids'])
            for i in range(len(input_ids[0])):
                if input_ids[0, i].item() in trouble_tokens:
                    input_ids[0, i] = tokenizer.unk_token_id
            input_ids = input_ids.to(DEVICE)
            attention_mask = torch.tensor(encoded['attention_mask'], device=DEVICE)
            length = input_ids.size()[1]
            if 1024 >= length > 512:
                output_part_0 = model(input_ids[:, :512], attention_mask=attention_mask[:, :512]).logits
                output_part_1 = model(input_ids[:, -512:], attention_mask=attention_mask[:, -512:]).logits
                overlapping_region = length - 2 * (length - 512)
                output = torch.hstack([output_part_0, output_part_1[:, overlapping_region:]])
                assert output.size() == (1, length, len(LABELS)), output.size()
            else:
                output = model(input_ids, attention_mask=attention_mask).logits
            labels = output.argmax(dim=2)
            total_string = ''
            current_label = 0
            current_word = ''
            tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids[0])
            display_lbl = ['', '.', '?', '!', ':', ';', ',', '-', '...']
            original_words = line.split(' ')
            original_words_idx = 0
            for token, label in zip(tokens, labels[0]):
                if not token.startswith('##'):
                    if token == '[UNK]':
                        assert original_words[original_words_idx] == current_word, "\n" + total_string + "\n" + line + "\n" + current_word + "\n" + \
                                                                                     original_words[original_words_idx]
                        token = original_words[original_words_idx + 1]
                    if token not in tokenizer.special_tokens_map.values():
                        if original_words[original_words_idx] == current_word:
                            if current_word[-1] in id_to_lbl:
                                total_string = total_string + current_word + ' '
                            else:
                                total_string = total_string + current_word + display_lbl[current_label] + ' '
                            original_words_idx += 1
                            current_word = token
                            current_label = label
                        else:
                            assert len(current_word) < len(original_words[ original_words_idx]), "\n" + total_string + "\n" + line + "\n" + current_word + "\n" + \
                                                                                     original_words[original_words_idx]
                            current_word += token
                            assert len(current_word) <= len(original_words[original_words_idx]), "\n" + total_string + "\n" + line + "\n" + current_word + "\n" + \
                                                                                      original_words[original_words_idx]

                else:
                    if current_word is not None:
                        current_word += token[len('##'):]
            assert original_words[original_words_idx] == current_word, "\n" + str(original_words_idx) + "\n" + str(
                len(original_words)) + "\norig=" + original_words[original_words_idx] + "\ncurr=" + current_word
            if current_word[-1] in id_to_lbl:
                total_string = total_string + current_word + ' '
            else:
                total_string = total_string + current_word + display_lbl[current_label] + ' '
            assert original_words_idx + 1 == len(original_words)
            total_string = total_string.strip()
            # total_string = fix_bert_spaces(total_string)
            print(line.strip())
            print(total_string)
            print()
            print(total_string, file=o)
