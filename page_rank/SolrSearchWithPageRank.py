import requests
import json
import torch
from tqdm import tqdm
import urllib
import flask

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# solr/bin/solr start -e cloud
SOLR_PORT = 8983

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

graph = torch.ones(len(products), len(products), device=DEVICE)

with open('references.tsv') as f:
    for line in tqdm(f, desc="Counting references", total=total):
        src, dst = line.split()
        src, dst = products[src], products[dst]
        graph[dst, src] += 1

graph.div_(graph.sum(dim=0).unsqueeze(0))
likelihoods = torch.ones(len(products), device=DEVICE)
print("iteration=0, sum=" + str(likelihoods.sum().item()) + ", non-zero elements=" + str(
    (likelihoods > 0).sum().item()))
for step in range(1, 100):
    likelihoods = graph @ likelihoods
    print("iteration=" + str(step) + ", sum=" + str(likelihoods.sum().item()) + ", non-zero elements=" + str(
        (likelihoods > 0).sum().item()))


def search(query):
    query = 'AND'.join(['"' + w + '"' for w in query.split()])
    query = 'facets:(' + query + ')'
    query = urllib.parse.quote(query)
    query = 'fl=id&q='+query+'&rows=99999&start=0'
    url = "http://localhost:" + str(SOLR_PORT) + "/solr/gettingstarted/select?" + query
    print(url)
    data = json.loads(requests.get(url).content)
    product_ids = [(product['id'], products.get(product['id'])) for product in data['response']['docs'] if
                   product['id'] in products]
    product_ids.sort(key=lambda x: likelihoods[x[1]], reverse=True)
    return [best[0] for best in product_ids[:10]]


app = flask.Flask(__name__)


@app.route('/search', methods=['GET'])
def api():
    return json.dumps(search(flask.request.args.get('q')))


@app.route('/', methods=['GET'])
def home():
    return """
<!doctype html>

<html lang="en">
<head>
  <meta charset="utf-8">

  <title>The HTML5 Herald</title>
  <meta name="description" content="The HTML5 Herald">
  <meta name="author" content="SitePoint">
  <script>
    async function search(q){
     console.log("Search: " + q)
     const response = await fetch('http://127.0.0.1:5000/search?q='+encodeURIComponent(q));
     const myJson = await response.json(); //extract JSON from the http response
     links.innerHTML = '';
     for(var i=0;i<myJson.length;i++){
        link = 'https://www.amazon.co.jp/dp/'+myJson[i]
        console.log(link)
        listItem = document.createElement('li');
        anchor = document.createElement('a');
        // Add the item text
        anchor.innerHTML = link;
        anchor.setAttribute('href', link);
        listItem.appendChild(anchor);
        // Add listItem to the listElement
        links.appendChild(listItem);
     }
    }
  </script>
  <style>
    .search-input-container{
        display: flex;
        max-width: 1100px;
        width: 100%;
        margin: 0 auto;
    }

    .search-input-container > textarea{
        height: 25px;
        line-height: 25px;
        font-size: 17px;
        color: #495057;
        resize: none;
        flex-grow: 1;
        border: none;
        padding: 5px 10px;
        outline: none;
        background-color: #f3f3f3;
        padding: 10px;
        border-top-left-radius: 15px;
        border-bottom-left-radius: 15px;
    }

    .search-input-container > button{
        cursor: pointer;
        background: none;
        border: none;
        cursor: pointer;
        border-radius: 2px;
        font-size: 14px;
        font-weight: 400;
        background-color: #000;
        border-top-right-radius: 15px;
        border-bottom-right-radius: 15px;
        color: #fff;
    }

    #links{
        list-style: none;
        max-width: 1100px;
        width: 100%;
        margin: 1rem auto;
        padding: 0;
    }

    #links > li{
        padding: 0.5rem 0;
    }

    #links > li > a{
        color: #495057;
    }
  </style>
</head>

<body>
  <div class="search-input-container">
    <textarea id="query"></textarea> <br/>
    <button onclick="search(query.value)">Search</button> <br/>
  </div>
  <ul id="links"></ul>
</body>
</html>
"""

app.run()
