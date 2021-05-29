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

graph = torch.zeros(len(products), len(products))

with open('references.tsv') as f:
    for line in tqdm(f, desc="Counting references", total=total):
        src, dst = line.split()
        src, dst = products[src], products[dst]
        graph[dst, src] += 1
graph = graph.to(DEVICE)

graph.div_(graph.sum(dim=0).clamp_min_(1).unsqueeze(0))
likelihoods = torch.ones(len(products), device=DEVICE)
print("iteration=0, sum=" + str(likelihoods.sum().item()) + ", non-zero elements=" + str(
        (likelihoods > 0).sum().item()))
for step in range(1, 100):
    likelihoods = graph @ likelihoods
    print("iteration=" + str(step) + ", sum=" + str(likelihoods.sum().item()) + ", non-zero elements=" + str(
        (likelihoods > 0).sum().item()))

# iteration=0, sum=27270.0, non-zero elements=27270
# iteration=1, sum=4661.0, non-zero elements=26995
# iteration=2, sum=1989.7718505859375, non-zero elements=26082
# iteration=3, sum=933.1207275390625, non-zero elements=24143
# iteration=4, sum=497.21661376953125, non-zero elements=23634
# iteration=5, sum=287.3778076171875, non-zero elements=23570
# iteration=6, sum=175.93458557128906, non-zero elements=23555
# iteration=7, sum=113.25997924804688, non-zero elements=23555
# iteration=8, sum=76.45773315429688, non-zero elements=23555
# iteration=9, sum=54.14463806152344, non-zero elements=23555
# iteration=10, sum=40.244049072265625, non-zero elements=23555
# iteration=11, sum=31.37636375427246, non-zero elements=23555
# iteration=12, sum=25.595375061035156, non-zero elements=23555
# iteration=13, sum=21.750377655029297, non-zero elements=23555
# iteration=14, sum=19.144786834716797, non-zero elements=23555
# iteration=15, sum=17.34804916381836, non-zero elements=23555
# iteration=16, sum=16.0888614654541, non-zero elements=23555
# iteration=17, sum=15.193110466003418, non-zero elements=23555
# iteration=18, sum=14.547100067138672, non-zero elements=23555
# iteration=19, sum=14.075347900390625, non-zero elements=23555
# iteration=20, sum=13.726933479309082, non-zero elements=23555
# ...
# iteration=60, sum=12.602633476257324, non-zero elements=19698
# ...
# iteration=97, sum=12.602435111999512, non-zero elements=16652
# iteration=98, sum=12.602435111999512, non-zero elements=16603
# iteration=99, sum=12.602435111999512, non-zero elements=16489
