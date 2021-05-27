# With help from tutorial
# https://huggingface.co/transformers/training.html
from transformers import BertForMaskedLM, BertTokenizer, pipeline, BertForTokenClassification, BertTokenizerFast
import re
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

##################################################################################
# Get data from https://github.com/poleval/2021-punctuation-restoration
##################################################################################

LABELS = {'.': 1, '?': 2, '!': 3, ':': 4, ';': 5, ',': 6, '-': 7, '...': 8, '': 0}
id_to_lbl = ['', '.', '?', '!', ':', ';', ',', '-', '...']
sep = re.compile('[ .?!:;,"-]+')
DATA_WORDS = []
DATA_LABELS = []
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

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = BertTokenizerFast.from_pretrained("dkleczek/bert-base-polish-cased-v1")
# nlp = pipeline('ner', model=model, tokenizer=tokenizer, device=DEVICE)

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
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = torch.tensor(doc_labels, dtype=torch.int)
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

model = BertForTokenClassification.from_pretrained("dkleczek/bert-base-polish-cased-v1", num_labels=len(LABELS))
model.to(DEVICE)
model.train()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()

# model = BertForMaskedLM.from_pretrained("dkleczek/bert-base-polish-cased-v1")
# tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-cased-v1")
# nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)
# for pred in nlp(f"Adam Mickiewicz wielkim polskim {nlp.tokenizer.mask_token} był."):
#     print(pred)
