from transformers import pipeline

# 1. First run any OCR tool on the pdf file and convert it to text.
# 2. Then take those sentences and replace gaps with bert mask.
# 3. Run BERT.
# 4. Profit! (homework done)

nlp = pipeline("fill-mask", model='bert-base-cased')
print(nlp(f"Starting your {nlp.tokenizer.mask_token} business could be the way to achieving"))
print(nlp(f"be the way to achieving financial independence, or {nlp.tokenizer.mask_token} could just as well land you in debt for the rest of your life"))
print(nlp(f"That, at {nlp.tokenizer.mask_token}, is the view of Charles and Brenda"))
print(nlp(f"a Scottish couple, who last week saw {nlp.tokenizer.mask_token} fish farm business put into the hands of the receiver"))
print(nlp(f"We started the business in 1985 when {nlp.tokenizer.mask_token} was being encouraged by the banks to borrow money"))
print(nlp(f"{nlp.tokenizer.mask_token} the time we were sure that we could make"))
print(nlp(f"the time we were sure that we could make {nlp.tokenizer.mask_token} into a going concern"))
print(nlp(f"said Charles Leggat, a farmer from {nlp.tokenizer.mask_token} Highlands"))
print(nlp(f"and the banks lent us more or less {nlp.tokenizer.mask_token} we asked for"))

