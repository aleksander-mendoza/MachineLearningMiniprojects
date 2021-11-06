import random
import urllib
from util import *
import selenium
from bs4 import BeautifulSoup
import bs4
import requests
import traceback
from fake_headers import Headers
from selenium import webdriver
import os
from tqdm import tqdm
import re
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from seleniumwire import webdriver

BY_BEST = 'https://my-best.com'
HEIM = 'https://heim.jp'
SAKIDORI = 'https://sakidori.co'
AMAZON = "https://www.amazon.co.jp"

header = Headers(
    headers=True  # generate misc headers
)


def get_header():
    h = header.generate()
    h['Accept-Language'] = 'jp-JP;q=0.5,jp;q=0.3'
    return h


def get_html(url):
    print(url)
    return requests.get(url, headers=get_header())


def download_html(url, file, add_comment_with_url=False):
    """
    :param url:
    :param add_comment_with_url: will prepend line with HTML comment containing source URL
    :param file:
    :return: True if downloaded file, False if file already existed
    """
    if os.path.isfile(file):
        return False
    res = get_html(url)
    with open(file, 'wb+') as f:
        if add_comment_with_url:
            f.write(('<!--' + url + '-->\n').encode('utf-8'))
        f.write(res.content)
    return True


def get_head(url):
    print(url)
    return requests.head(url, headers=get_header(), allow_redirects=True)


def get_soup(url):
    return BeautifulSoup(get_html(url).content, 'html.parser')


def my_best_get_category_page(category, page):
    return get_soup('https://my-best.com/presses?category=' + str(category) + '&page=' + str(page) + '&type=research')


def my_best_get_article_lists(category, page):
    return get_soup('https://my-best.com/lists?category=' + str(category) + '&page=' + str(page))


def my_best_get_article_list(article_list_no):
    return get_soup('https://my-best.com/lists/' + str(article_list_no))


def my_best_scrape_article_lists(my_best_categories, category, page):
    soup = my_best_get_article_lists(category, page)
    page_bar = soup.select('ul.c-pagination li a')
    last_page = int(page_bar[-1].text) if page_bar else -1

    for article in soup.select('ul.p-item-card li a'):
        url = article['href']
        assert url.startswith("/lists/"), url
        article_list_no = int(url[len("/lists/"):])
        heading = article.select('h2.p-item-card__heading')[0].text
        my_best_categories[category][2][article_list_no] = heading
    return last_page > page


def my_best_scrape_article_lists_all_pages(my_best_categories, category):
    page = 1
    print("Page " + str(page))
    while my_best_scrape_article_lists(my_best_categories, category, page):
        page += 1
        print("Page " + str(page))


def my_best_scrape_all_article_lists_all_pages(my_best_categories):
    for category, (title, subtitle, articles) in my_best_categories.items():
        print("Category " + str(category))
        if len(articles) == 0:
            my_best_scrape_article_lists_all_pages(my_best_categories, category)


def my_best_scrape_all_article_lists_all_pages_and_save():
    my_best_categories = load_json('my_best_categories.json')
    try:
        my_best_scrape_all_article_lists_all_pages(my_best_categories)
    except Exception as e:
        traceback.print_exc()
    save_json(my_best_categories, 'my_best_categories.json')


def my_best_scrape_product_list(articles_collection, article_list_no):
    soup = my_best_get_article_list(article_list_no)
    products = []
    for product in soup.select('div.c-item-list__wrapper div.js-parts'):
        heading = product.select('h4.c-heading-catch')
        heading = heading[0].text if heading else ''
        description = product.select('div.p-wysiwyg')[0].text
        title = product.select('h3.c-panel__heading')[0]
        if len(title.contents) > 1 and type(title.contents[0]) == bs4.element.Tag:
            producer = title.contents[0].text
            product_name = title.contents[1]
        else:
            producer = ''
            product_name = title.text
        hrefs = [button['href'] for button in product.select('a.c-shops__button')]
        products.append((producer, product_name, heading, hrefs, description))
    articles_collection[article_list_no] = products


def my_best_scrape_all_products(my_best_categories, articles_collection):
    for category, (title, subtitle, articles) in my_best_categories.items():
        for article_list_no, article_title in articles.items():
            if article_list_no not in articles_collection:
                my_best_scrape_product_list(articles_collection, article_list_no)


def my_best_scrape_all_products_and_save():
    my_best_categories = load_json('my_best_categories.json')
    articles_collection = load_json('my_best_articles.json')
    try:
        my_best_scrape_all_products(my_best_categories, articles_collection)
    except Exception as e:
        traceback.print_exc()

    save_json(articles_collection, 'my_best_articles.json')


AFFILIATE_ONLINE_STORES = ['www.amazon.co.jp', 'amazon.co.jp', 'shopping.yahoo.co.jp', 'item.rakuten.co.jp',
                           'search.rakuten.co.jp', 'www.nitori-net.jp', 'm.ikea.com', 'www.muji.net', 'bande.ne.jp',
                           'anbas.cart.fc2.com', 'www.bathlier.jp', 'www.dinos.co.jp', 'lohaco.jp', 'sanyo-i.jp',
                           'www.le-noble.com', 'www.wepkyoto.co.jp', 'www.importshopaqua.com',
                           'store.shopping.yahoo.co.jp', 'paypaymall.yahoo.co.jp', 'liflance.jp', 'webshop.montbell.jp',
                           'jp.globe-trotter.com', 'loft.omni7.jp', 'sunshine-cloud.com', 'www.carl.co.jp',
                           'www.jj-craft.com', 'www.carton-opt.net', 'craft.lab.craypas.com', 'www.releaf.co.jp',
                           'www.manza.co.jp', 'www.roboform.com', 'www.e-primal.com', 'www.amazon.com',
                           'shop.extended.jp', 'store.leica-camera.jp', 'sachiopia.base.ec', 'www.mtgec.jp',
                           'www.muji.com', 'www.snola.co.jp', 'www.ecolecriollo.co.jp', 'jp.iherb.com',
                           'aidenchoco.official.ec', 'beauty-architect.com', 'ijkhair.buyshop.jp',
                           'www.sanct-japan.co.jp', 'otafukudo.com', 'www.herbteas.shop', 'www.nealsyard.co.jp',
                           'www.1protein.com', 'drop-tree-of-life.com', 'www.aujua.com', 'pb.nidrug.co.jp',
                           'www.serendibeauty.jp', 'm.jp.lavien.co.kr', 'www.qoo10.jp', 'www.cycleplus.jp',
                           'www.fancl.co.jp', 'www.mimc.co.jp', 'www.cosmekitchen-webstore.jp', 'www.paraube.com',
                           'celvoke.com', 'toonecosmetics.com', 'sincere-garden.jp', 'sarakuwa.shop',
                           'www.lavera.co.jp', 'www.tv-movie.co.jp', 'jurlique-japan.com', 'slowbliss.jp',
                           'f-organics.jp', 'www.lancome.jp', 'etvos.com', 'lagomcosmetics.jp',
                           'www.peaceofshine.co.jp', 'a-c-p.tokyo', 'www.revolveclothing.co.jp', 'www.cutepress.jp',
                           'www.yslb.jp', 'shiro-shiro.jp', 'www.dior.com', 'www.shuuemura.jp', 'www.cando-web.co.jp',
                           'meeco.mistore.jp', 'www.qoo10.com', 'imvely.jp', 'vtcosmetics.jp', 'www.buyma.com',
                           'www.kracie-salon.com', 'www.isehan.co.jp', 'www.maccosmetics.jp', 'www.cosme.com',
                           'www.riceforce.com', 'www.clinique.jp', 'www.makeupforever.jp', 'shop.conscious.co.jp',
                           'femininestyle.jp', 'eslucy.com', 'moreravi.jp', 'yoridori-dobest.jp',
                           'www.lookfantastic.jp', 'atelierraisin.cart.fc2.com', 'maonail.jp', 'www.felissimo.co.jp',
                           'www.bellemaison.jp', 'shanzenet.base.shop', 'www.ichigo-yakusoku.com', 'www.kirei-cosme.jp',
                           'asafuku.jp', 'music.apple.com', 'www.narscosmetics.jp', 'brightage.jp',
                           'prd-v3-i.chanel.com', 'www.mebiusseiyaku.co.jp', 'www.emi-net.co.jp', 'jn.lush.com',
                           'bs-cosme.com', 'prioricosme.net', 'www.dr-recella.com', 'avancia.co.jp',
                           'www.cuore-onlineshop.jp', 'vitalmaterial.com', 'www.plantologyus.com', 'perfumeoil.co.jp',
                           'www.sawaicoffee.net', 'iwateya-shop.jp', 'yuzusco.com', 'shop.henko.co.jp', 'ec.gazta.jp',
                           'shop.cake-cake.net', 'givreetokyo.shop-pro.jp', 'www.akanean-shop.com',
                           'www.ukai-online.com', 'www.hokuei-foods.com', 'kite-misawa.com', 'curryland.theshop.jp',
                           'www.konpeito.co.jp', 'www.matsukiyo.co.jp', 'www.topvalu.net', 'www.chateraise.co.jp',
                           'www.ofk-ec.com', 'www.gion-murata.co.jp', 'beillevaire.jp', 'fresco.buyshop.jp',
                           'www.sakaiseimen.com', 'www.cantipasto.biz', 'store.kinoya.co.jp', 'www.bbck.jp',
                           'www.kuradashi-yakiimo.com', 'katogyu.co.jp', 'tontachi.stores.jp', 'shop.asahiya.net',
                           'www.ginza-yoshizawa.com', 'www.shinise.ne.jp', 'sangosho.shop35.makeshop.jp',
                           'www.picard-frozen.jp', 'tsubota.buyshop.jp', 'miyatameat.shop-pro.jp', 'koushindo.net',
                           'coconutdreambakery.com', 'www.vapeur1160.com', 'www.sentaro.co.jp', 'www.jeremyjemimah.com',
                           'www.8284.co.jp', 'sino.shop38.makeshop.jp', 'eian-shop.com', 'murata-shop.jp',
                           'solco.stores.jp', 'nishikiya-shop.com', 'www.majimena-hachimitsu.com', 'kei-ei.com',
                           'www.enchan-the.com', 'kaneroku-matsumotoen.easy-myshop.jp', 'www.pomshop.jp',
                           'www.takashimaya.co.jp', 'www.benoist.co.jp', 'shop.grape-republic.com',
                           'rec-coffee.stores.jp', 'www2.enekoshop.jp', 'kabuki-store.com', 'ciel-blue.jp',
                           'likaman.net', 'likaman.co.jp', 'www.biokura.jp', 'matsuzaki-lemongrass.com',
                           'www.cuoca.com', 'www.conranshop.jp', 'www.amepla.jp', 'www.ikea.com', 'www.postdetail.com',
                           'magewappa.com', 'www.utsuwa-hanada.jp', 'www.tujiwa-kanaami.com', 'www.zakkaworks.com',
                           'store.deandeluca.co.jp', 'www.saga-city.jp', 'kawasaki-plastics.jp', 'www.balmuda.com',
                           'www.villeroy-boch.co.jp', 'www.zarahome.com', 'kougeitakumi.shop28.makeshop.jp',
                           'uchill.jp', 'atc.official.ec', 'nishiyama-shop.stores.jp', 'partymarket365.stores.jp',
                           'butterdrop.shop-pro.jp', 'myglassplate.shop-pro.jp', 'www.mydear-life.com',
                           'jp.zwilling-shop.com', 'triumph-cpn.com', 'store.fujibo-ap.jp', 'www.uniqlo.com',
                           'www.a-trend-ld-store.net', 'jp.triumph.com', '5-fifth.com', 'www.danielwellington.com',
                           'jp.chicwish.com', 'www.privatelabo.jp', 'totb.stores.jp', 'www.joint-space.co.jp']


def url_strip_protocol(url):
    if url.startswith('https://'):
        return url[len('https://'):]
    if url.startswith('http://'):
        return url[len('http://'):]
    return url


def amazon_normalize_url(url):
    if url.startswith(AMAZON):
        url = url[len(AMAZON):]
    url = url.strip()
    url = url.split('?')[0]
    url = url.split('ref=')[0]
    return url


def is_affiliate_online_store(url):
    url = url_strip_protocol(url)
    for store in AFFILIATE_ONLINE_STORES:
        if url.startswith(store):
            return True
    return False


def get_url_query_param(param_name, url):
    parts = url.split('?')
    if len(parts) > 1:
        parts = parts[1].split('&')
        param_name = param_name + '='
        for part in parts:
            if part.startswith(param_name):
                return urllib.parse.unquote(part[len(param_name):])
    else:
        return None


def my_best_resolve_redirects(articles_collection, resolve_redirects_with_selenium=True):
    if resolve_redirects_with_selenium:
        driver = webdriver.Firefox()
    try:
        for article_list_no, article in tqdm(articles_collection.items(), desc="articles processed"):
            for (producer, product_name, heading, hrefs, description) in article:
                for i in range(len(hrefs)):
                    href = hrefs[i]
                    if href.startswith("http://"):
                        href = "https://" + href[len("http://"):]
                    href_no_proto = url_strip_protocol(href)
                    if href_no_proto.startswith('www.amazon.co.jp') and ('?' in href or '/ref=' in href):
                        hrefs[i] = amazon_normalize_url(href)
                        # print(href + "\n-amazon->\n" + hrefs[i])
                    elif href_no_proto.startswith('ck.jp.ap.valuecommerce.com'):
                        hrefs[i] = get_url_query_param('vc_url', href)
                        # print(href + "\n-ck.jp->\n" + hrefs[i])
                    elif href_no_proto.startswith('hb.afl.rakuten.co.jp'):
                        hrefs[i] = get_url_query_param('m', href)
                        # print(href + "\n-rakuten->\n" + hrefs[i])
                    elif not is_affiliate_online_store(href):
                        if resolve_redirects_with_selenium:
                            print(href)
                            try:
                                driver.get(href)
                            except selenium.common.exceptions.WebDriverException as e:
                                if 'dnsNotFound' in e.msg:
                                    hrefs[i] = ''
                                    print("Dead link")
                                else:
                                    traceback.print_exc()
                                continue
                            wait = WebDriverWait(driver, 10)
                            try:
                                wait.until(EC.url_changes(href))
                                hrefs[i] = driver.current_url
                                print("--> " + hrefs[i])
                            except selenium.common.exceptions.TimeoutException as e:
                                print("No redirect")
                                AFFILIATE_ONLINE_STORES.append(href_no_proto.split('/')[0])
    except Exception as e:
        traceback.print_exc()
    save_json(articles_collection, 'my_best_articles.json')
    print(AFFILIATE_ONLINE_STORES)
    if resolve_redirects_with_selenium:
        driver.close()


def my_best_extract_amazon_products(articles_collection):
    amazon_product_ids = []
    for article_list_no, article in tqdm(articles_collection.items(), desc="articles processed"):
        for (producer, product_name, heading, hrefs, description) in article:
            for href in hrefs:
                if href.startswith(AMAZON):
                    product_id = amazon_get_product_from_link(href)
                    if product_id:
                        amazon_product_ids.append(product_id)
    return amazon_product_ids


AMAZON_PRODUCT_ID_PARSER = re.compile("/dp/(.*?)/")


def amazon_get_product_from_link(url):
    match = AMAZON_PRODUCT_ID_PARSER.search(url)
    if match:
        return match.group(1)
    return None


def amazon_get_link_from_product(product_id):
    return 'https://www.amazon.co.jp/dp/' + product_id + '/'


def amazon_download_product_page(product_id):
    return download_html(amazon_get_link_from_product(product_id), 'amazon_products/' + product_id + '.html')


def amazon_download_all_pages_of_product_ids(product_ids):
    for i, product_id in enumerate(tqdm(product_ids, desc="products processed")):
        if amazon_download_product_page(product_id):
            time.sleep(random.randint(0, i % 10))
        # Sleeping is necessary so that amazon does detect that we are a bot.
        # Otherwise it will block us with captchas


def amazon_download_all_pages_of_product_ids_extracted_from_my_best(articles_collection):
    amazon_download_all_pages_of_product_ids(my_best_extract_amazon_products(articles_collection))


def amazon_is_captcha_page(page_contents):
    return 'To discuss automated access to Amazon data please contact' in page_contents


def amazon_is_error_page(page_contents):
    return "We can't connect to the server for this app or website" in page_contents


def amazon_is_invalid_url_page(page_contents):
    return "Invalid URL" in page_contents


def amazon_filter_pages(filter_fn):
    product_ids = []
    for file in tqdm(os.listdir('amazon_products'), desc='files scanned'):
        product_id = file[:-len('.html')]
        with open('amazon_products/' + file, 'r', encoding='utf-8', errors='replace') as fd:
            page_contents = fd.read()
        if filter_fn(page_contents):
            product_ids.append(product_id)
    return product_ids


def amazon_find_captchas():
    return amazon_filter_pages(amazon_is_captcha_page)


FIREFOX_DRIVER = None
FIREFOX_HEADER = get_header()


def get_firefox_driver():
    global FIREFOX_DRIVER
    if FIREFOX_DRIVER is None:
        FIREFOX_DRIVER = webdriver.Firefox()

        def interceptor(request):
            for k in request.headers.keys():
                del request.headers[k]
            for k, v in FIREFOX_HEADER.items():
                request.headers[k] = v

        FIREFOX_DRIVER.request_interceptor = interceptor
    return FIREFOX_DRIVER


def amazon_download_and_handle_captchas(product_ids):
    driver = get_firefox_driver()
    for product_id in tqdm(product_ids, desc="products processed"):
        file = 'amazon_products/' + product_id + '.html'
        if os.path.isfile(file):
            with open(file, encoding='utf-8', errors='replace') as fd:
                page_contents = fd.read()
            if amazon_is_captcha_page(page_contents) \
                    or amazon_is_error_page(page_contents) \
                    or amazon_is_invalid_url_page(page_contents):
                pass
            else:
                continue
        url = amazon_get_link_from_product(product_id)
        driver.get(url)
        if amazon_is_captcha_page(driver.page_source):
            print("Page for " + product_id + " is captcha after redownload")
            try:
                input_field = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'captchacharacters')))
                captcha = input("What's the captcha?")
            except selenium.common.exceptions.TimeoutException as e:
                traceback.print_exc()
                captcha = ''
                input_field = None
            if captcha != '':
                input_field.send_keys(captcha)
                driver.find_element_by_css_selector('button').click()
            with open(file, 'w+') as f:
                f.write(driver.page_source)
        elif amazon_is_error_page(driver.page_source):
            print("Too much traffic " + product_id)
        elif amazon_is_invalid_url_page(driver.page_source):
            print("Invalid URL " + product_id)
        else:
            with open(file, 'w+') as f:
                f.write(driver.page_source)
            if not driver.find_elements_by_css_selector('#productTitle'):
                print("Incorrect page for " + product_id)
        time.sleep(10 + random.randint(0, 10))


def amazon_list_all_downloaded_products():
    return [file[:-len('.html')] for file in os.listdir('amazon_products')]


def amazon_scrape_deeper_level_links(current_level_product_ids):
    deeper_level_product_ids = []
    all_levels_product_ids = amazon_list_all_downloaded_products()
    for product_id in tqdm(current_level_product_ids, desc="scraped files"):
        with open('amazon_products/' + product_id + '.html', 'r', encoding='utf-8', errors='replace') as fd:
            page_contents = fd.read()
        soup = BeautifulSoup(page_contents, 'html.parser')
        for a in soup.select('a[href]'):
            new_product_id = amazon_get_product_from_link(a['href'])
            if new_product_id and new_product_id not in all_levels_product_ids:
                deeper_level_product_ids.append(new_product_id)
    return deeper_level_product_ids


def amazon_scrape_next_level_links_and_save():
    current_level_product_ids = load_json('amazon_current_level_products.json')
    current_level_product_ids = amazon_scrape_deeper_level_links(current_level_product_ids)
    save_json(current_level_product_ids, 'amazon_current_level_products.json')


def amazon_download_current_level_products():
    current_level_product_ids = load_json('amazon_current_level_products.json')
    amazon_download_and_handle_captchas(current_level_product_ids)


def amazon_extract_facets(facet_consumer):
    for file in tqdm(os.listdir('amazon_products'), desc="files scraped for facets"):
        product_id = file[:-len('.html')]
        try:
            with open('amazon_products/' + file) as fd:
                soup = BeautifulSoup(fd, 'html.parser')
                bullets = soup.select('#feature-bullets ul li span')
                if not bullets:
                    feat_bullets = soup.select('#feature-bullets')
                    assert len(feat_bullets) == 0 or \
                           len(feat_bullets[0].contents) == 0 or \
                           (len(feat_bullets[0].contents) == 1 and feat_bullets[0].contents[0].strip() == '')
                facets = [bullet.text.strip() for bullet in bullets]
                bullets = soup.select('#detailBullets_feature_div ul li span')
                if not bullets:
                    assert not soup.select('#detailBullets_feature_div')
                facets += [bullet.text.strip() for bullet in bullets]
                bullets = soup.select('#poExpander > div > div > table tr')
                if not bullets:
                    assert not soup.select('#poExpander')
                for bullet in bullets:
                    name, value = bullet.select('td span')[:2]
                    facets.append((name.text.strip(), value.text.strip()))
                bullets = soup.select('#productDetails_techSpec_section_1 tr')
                if not bullets:
                    assert not soup.select('#productDetails_techSpec_section_1')
                for bullet in bullets:
                    name = bullet.select('th')[0]
                    value = bullet.select('td')[0]
                    facets.append((name.text.strip(), value.text.strip()))
            facet_consumer(product_id, facets)
        except Exception as e:
            traceback.print_exc()
            print(product_id + " failed")


def amazon_extract_unique_facets():
    unique_facets = {}

    def add_facet(prod_id, facets):
        for facet in facets:
            if type(facet) == tuple:
                key, value = facet
                if key == 'カラー' or key == 'colour' or key == 'color' or key == 'Colour' or key == 'Color':
                    key = '色'
                elif key == 'ブランド名':
                    key = 'ブランド'
                elif key == '素材':
                    key = '材質'
                elif key == '商品の重量' or key == '商品重量':
                    key = '重量'
                values_and_ids = unique_facets.get(key)
                if values_and_ids is not None:
                    prod_ids = values_and_ids.get(value)
                    if prod_ids is not None:
                        prod_ids.append(prod_id)
                    else:
                        values_and_ids[value] = [prod_id]
                else:
                    unique_facets[key] = {value: [prod_id]}

    amazon_extract_facets(add_facet)
    return unique_facets


def amazon_extract_facets_and_save():
    all_docs = amazon_extract_unique_facets()
    save_json(all_docs, 'amazon_facets.json')


def amazon_redownload_captchas():
    amazon_download_and_handle_captchas(amazon_find_captchas())


def amazon_extract_titles_and_categories():
    title_and_categories = []
    for file in tqdm(os.listdir('amazon_products'), desc="files scraped for titles and categories"):
        product_id = file[:-len('.html')]
        try:
            with open('amazon_products/' + file) as fd:
                soup = BeautifulSoup(fd, 'html.parser')
                categories = soup.select('#wayfinding-breadcrumbs_feature_div li a')
                if categories:
                    categories = categories[-1].text.strip()
                else:
                    categories = soup.select('#nav-subnav')[0].text.strip()
                productTitle = soup.select('#productTitle')[0]
                productSubtitle = soup.select('#productSubtitle')
                if productSubtitle:
                    title = productTitle.text.strip() + ' ' + productSubtitle[0].text.strip()
                else:
                    title = productTitle.text.strip()
                title_and_categories.append((title, categories))
        except Exception as e:
            traceback.print_exc()
            print(product_id + " failed")
    with open('data.tsv', 'w+') as f:
        for title, category in title_and_categories:
            f.write(category+'\t'+title+'\n')



amazon_extract_titles_and_categories()
