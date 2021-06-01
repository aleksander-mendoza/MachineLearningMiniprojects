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

with open('facets.xml', 'w+') as f:
    print("<add>", file=f)
    for doc, facets in DOCS.items():
        print("  <doc>", file=f)
        print("    <field name=\"id\">" + doc + "</field>", file=f)
        print("    <field name=\"facets\">"+" ".join([escape(facet)+" "+escape(facet_val) for facet, facet_val in facets.items()])+"</field>", file=f)
        print("  </doc>", file=f)

# Command to post to solr
# solr/bin/post -c YOUR_COLLECTION facets.xml
