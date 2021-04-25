from bs4 import BeautifulSoup
import requests
import re
import traceback
import os
from collections import deque
import select
import sys

AMAZON_URL = "https://www.amazon.co.jp/"

URL_DB = {}
URL_STACK = deque()
checkpoint = open('checkpoint.txt', 'a+')


def push_link(url):
    url = url.strip()
    if url.startswith("/") \
            and not url.startswith("/gp/") \
            and not url.startswith("/hz/") \
            and not url.startswith("/pcr/") \
            and not url.startswith("/slp/") \
            and not url.startswith("/ranking") \
            and not url.startswith("/gcx/") \
            and not url.startswith("/deal/") \
            and not url.startswith("/ask/") \
            and not url.startswith("/stores/") \
            and not url.startswith("/ref=") \
            and '/product-reviews/' not in url \
            and '/review/' not in url \
            and '/b/' not in url \
            and '/b?' not in url:
        if url not in URL_DB:
            URL_STACK.append(url)
            URL_DB[url] = {}
    else:
        pass
        # print("Filtered " + AMAZON_URL + url)


MAC = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36'
UBUNTU = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:86.0) Gecko/20100101 Firefox/86.0"


def get_url(url):
    print(url)
    resp = requests.get(url, headers={
        'user-agent': UBUNTU,
        # 'sec-ch-ua': '"Google Chrome";v="89", "Chromium";v="89", ";Not A Brand";v="99"'
    })
    return BeautifulSoup(resp.content, 'html.parser')


WS = re.compile('\\s+')


def norm(s):
    return WS.sub(' ', s.strip())


def write_data(url, data, fd, flush=False):
    fd.write(url)
    fd.write('\t')
    fd.write(data['title'])
    fd.write('\t')
    fd.write(data['features'])
    fd.write('\t')
    fd.write(';'.join(data['categories']))
    fd.write('\n')
    if flush:
        fd.flush()


def product_page():
    url = URL_STACK.popleft()
    soup = get_url(AMAZON_URL + url)
    title = soup.select("#productTitle")
    if title:
        title = title[0]
        data = URL_DB[url]
        data['title'] = norm(title.text)
        # byline = soup.select("#bylineInfo_feature_div")
        # if byline:
        #     byline = byline[0]
        #     byline_url = byline.select("a[href]")
        #     data['byline'] = byline.text.strip()
        #     if byline_url:
        #         byline_url = [x['href'] for x in byline_url]
        #         data['byline_url'] = byline_url
        # price = soup.select("#priceblock_ourprice")
        # if price:
        #     data['price'] = price[0].text.strip()
        # else:
        #     price = soup.select("#newBuyBoxPrice")
        #     if price:
        #         data['price'] = price[0].text.strip()
        features = soup.select("#feature-bullets")
        if features:
            data['features'] = norm(features[0].text)
        else:
            data['features'] = ''
        # about = soup.select("#aplus")
        # img = soup.select("#landingImage")
        # if img:
        #     data['img_url'] = img[0]['src']
        category = soup.select("#wayfinding-breadcrumbs_feature_div li a")
        data['categories'] = [norm(c.text) for c in category]
        print("Updated ", data)
        write_data(url, data, checkpoint, True)
        for deeper_url in soup.select("a[href]"):
            push_link(deeper_url["href"])
    else:
        print("Skipping")
    return url


if os.path.isfile('db.tsv') and os.path.isfile('stack.txt'):
    with open('db.tsv', 'r') as db:
        for line in db:
            page_url, page_title, page_features, page_categories = line.split('\t')
            URL_DB[page_url] = {}
    with open('stack.txt', 'r') as stack:
        for page_url in stack:
            page_url = page_url.strip()
            URL_STACK.append(page_url)
            URL_DB[page_url] = {}
else:
    page = get_url(AMAZON_URL)
    for link in page.select("a[href]"):
        push_link(link["href"])

for i in range(5000):
    if i % 10 == 0:
        i, _, _ = select.select([sys.stdin], [], [], 0)
        if i:
            _ = sys.stdin.readline().strip()
            break
    try:
        last_page = product_page()
    except Exception as e:
        traceback.print_exc()
        break

with open('stack.txt', 'w+') as stack:
    stack.write(last_page)
    stack.write('\n')
    for last_page in URL_STACK:
        stack.write(last_page)
        stack.write('\n')

with open('db.tsv', 'a+') as db:
    for page_url, page_data in URL_DB.items():
        if len(page_data) > 0:
            write_data(page_url, page_data, db)
