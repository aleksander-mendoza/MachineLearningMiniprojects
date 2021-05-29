import torch
from tqdm import tqdm

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

torch.autograd.set_grad_enabled(False)

products = set()
total = 0
with open('references.tsv') as f:
    for line in tqdm(f, desc="Preliminary scan"):
        src, dst = line.split()
        products.add(src)
        products.add(dst)
        total += 1

products = {prod_id: idx for idx, prod_id in enumerate(products)}

graph = torch.ones(len(products), len(products))

with open('references.tsv') as f:
    for line in tqdm(f, desc="Counting references", total=total):
        src, dst = line.split()
        src, dst = products[src], products[dst]
        graph[dst, src] += 1
graph = graph.to(DEVICE)

graph.div_(graph.sum(dim=0).unsqueeze(0))
likelihoods = torch.ones(len(products), device=DEVICE)
print("iteration=0, sum=" + str(likelihoods.sum().item()) + ", non-zero elements=" + str(
        (likelihoods > 0).sum().item()))
for step in range(1, 100):
    likelihoods = graph @ likelihoods
    print("iteration=" + str(step) + ", sum=" + str(likelihoods.sum().item()) + ", non-zero elements=" + str(
        (likelihoods > 0).sum().item()))

# iteration=0, sum=27270.0, non-zero elements=27270
# iteration=1, sum=27270.01171875, non-zero elements=27270
# ...
# iteration=96, sum=27270.15234375, non-zero elements=27270
# iteration=97, sum=27270.15234375, non-zero elements=27270
# iteration=98, sum=27270.15234375, non-zero elements=27270
# iteration=99, sum=27270.15234375, non-zero elements=27270
