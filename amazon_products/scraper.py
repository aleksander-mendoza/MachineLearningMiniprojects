import os
import bs4

for file in os.listdir('html'):
    with open("html/" + file) as fp:
        soup = bs4.BeautifulSoup(fp, 'html.parser')
        soup.select('#prodDetails')
        soup.select('#productDescription')
        soup.select('#feature-bullets')
        soup.select('#productTitle')

