import sys
import traceback
import os
from bs4 import BeautifulSoup

AMAZON_URL = "https://www.amazon.co.jp/"
URLS = set()
titles = {}
with open('db.tsv', 'r') as db:
    for url_and_file in db:
        url, file = url_and_file.split(' ')
        if url in URLS:
            print("Duplicate " + url)
        URLS.add(url)
        file = file.strip()
        path = 'html/' + file + ".html"
        if os.path.isfile(path):
            with open(path, errors='replace') as h:
                try:
                    link = h.readline()
                    link = link[len("<!--"):-len("-->\n")]
                    assert link == url, link + " != " + url + " for index " + file
                    content = h.read()
                    soup = BeautifulSoup(content)
                    title = soup.select('#productTitle')
                    if title:
                        title = title[0]
                        if title in titles:
                            print('Duplicate title in '+path + " and "+titles[title])
                        else:
                            titles[title] = path
                except Exception as e:
                    traceback.print_exc()
                    print(file)
        else:
            print("Not exists " + path)

# /%E3%80%8E%E3%82%A6%E3%83%9E%E7%AE%B1%E3%80%8F%E7%AC%AC1%E3%82%B3%E3%83%BC%E3%83%8A%E3%83%BC-%E3%82%A2%E3%83%8B%E3%83%A1%E3%80%8E%E3%82%A6%E3%83%9E%E5%A8%98-%E3%83%97%E3%83%AA%E3%83%86%E3%82%A3%E3%83%BC%E3%83%80%E3%83%BC%E3%83%93%E3%83%BC%E3%80%8F%E3%83%88%E3%83%AC%E3%83%BC%E3%83%8A%E3%83%BC%E3%82%BABOX-Blu-ray-%E5%92%8C%E6%B0%A3%E3%81%82%E3%81%9A%E6%9C%AA/dp/B07BPYN93K/ref=pd_sim_1?pd_rd_w=5BvaA&pf_rd_p=1ebaba54-a282-454a-bd33-07860adf783c&pf_rd_r=HZEC8ND76ZJ64570WHE7&pd_rd_r=5779d960-dea4-426e-baec-69bfc6bfd774&pd_rd_wg=kKDem&pd_rd_i=B07BPYN93K&psc=1
# /%E3%80%8E%E3%82%A6%E3%83%9E%E7%AE%B1%E3%80%8F%E7%AC%AC1%E3%82%B3%E3%83%BC%E3%83%8A%E3%83%BC-%E3%82%A2%E3%83%8B%E3%83%A1%E3%80%8E%E3%82%A6%E3%83%9E%E5%A8%98-%E3%83%97%E3%83%AA%E3%83%86%E3%82%A3%E3%83%BC%E3%83%80%E3%83%BC%E3%83%93%E3%83%BC%E3%80%8F%E3%83%88%E3%83%AC%E3%83%BC%E3%83%8A%E3%83%BC%E3%82%BABOX-Blu-ray-%E5%92%8C%E6%B0%A3%E3%81%82%E3%81%9A%E6%9C%AA/dp/B07BPYN93K/ref=pd_bxgy_3/357-1131557-8696245?_encoding=UTF8&pd_rd_i=B07BPYN93K&pd_rd_r=8f2c6b36-5384-4a3a-ab75-2f356c6a7aee&pd_rd_w=JGUPd&pd_rd_wg=Q9GRl&pf_rd_p=105d6769-bacf-43d4-85ea-a25cec858a3c&pf_rd_r=MD7NNMFMWCKQY445NNK1&psc=1&refRID=MD7NNMFMWCKQY445NNK1