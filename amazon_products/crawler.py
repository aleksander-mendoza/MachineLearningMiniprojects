from bs4 import BeautifulSoup
import requests
import re
import traceback
import os
from collections import deque
import select
import sys
from fake_headers import Headers
import base64

header = Headers(
    headers=True  # generate misc headers
)

# from random_user_agent.user_agent import UserAgent
# from random_user_agent.params import SoftwareName, OperatingSystem, HardwareType
# user_agent_rotator = UserAgent(software_names=[e.value for e in SoftwareName],
#                                operating_systems=[e.value for e in OperatingSystem],
#                                hardware_type=[e.value for e in HardwareType], limit=500)


AMAZON_URL = "https://www.amazon.co.jp/"

MAX_INDEX = -1
URL_DB = {}
URL_STACK = deque()


def normalize_url(url):
    url = url.strip()
    url = url.split('?')[0]
    url = url.split('ref=')[0]
    return url


def push_link(url):
    url = normalize_url(url)
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
            and '/e/' not in url \
            and '/b?' not in url:
        if url not in URL_DB:
            URL_STACK.append(url)
            URL_DB[url] = -1
    else:
        pass
        # print("Filtered " + AMAZON_URL + url)


def next_page():
    global MAX_INDEX
    url = URL_STACK.popleft()
    print(AMAZON_URL + url)
    h = header.generate()
    h['Accept-Language'] = 'jp-JP;q=0.5,jp;q=0.3'
    resp = requests.get(AMAZON_URL + url, headers=h)
    soup = BeautifulSoup(resp.content, 'html.parser')
    MAX_INDEX += 1
    URL_DB[url] = MAX_INDEX
    f_path = 'html/' + str(MAX_INDEX) + '.html'
    assert not os.path.isfile(f_path)
    with open(f_path, 'wb+') as f:
        f.write(('<!--' + url + '-->\n').encode('utf-8'))
        f.write(resp.content)
    for deeper_url in soup.select("a[href]"):
        push_link(deeper_url["href"])
    return url


def save_stack():
    with open('stack.txt', 'w+') as stack:
        for last_page in URL_STACK:
            stack.write(last_page)
            stack.write('\n')
    with open('db.tsv', 'w+') as db:
        for url, file in URL_DB.items():
            if file != -1:
                db.write(url + ' ' + str(file) + '\n')


def load_stack():
    global MAX_INDEX
    with open('db.tsv', 'r') as db:
        for url_and_file in db:
            url, file = url_and_file.split(' ')
            file = int(file)
            URL_DB[url] = file
            if file > MAX_INDEX:
                MAX_INDEX = file
            else:
                assert file != MAX_INDEX, file + " " + MAX_INDEX
    with open('stack.txt', 'r') as stack:
        for page_url in stack:
            page_url = page_url.strip()
            URL_STACK.append(page_url)


load_stack()
try:
    i = 1
    while True:
        if i % 10 == 0:
            ii, _, _ = select.select([sys.stdin], [], [], 0)
            if ii:
                _ = sys.stdin.readline().strip()
                break
        if i % 100 == 0:
            save_stack()
        next_page()
        i += 1
except Exception as e:
    traceback.print_exc()

save_stack()
