import requests
import traceback
import json
from xml.sax.saxutils import escape
DOCS = {}
with open('amazon_facets.json') as f:
    facets = json.load(f)
    for facet, values in facets.items():
        for value, docs in values.items():
            for doc in docs:
                if doc not in DOCS:
                    DOCS[doc] = {}
                DOCS[doc][facet] = value

