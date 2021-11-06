from util import *
import math
from tqdm import tqdm


# MY_BEST_CATEGORIES = load_json('my_best_categories.json')

# ARTICLES = load_json('my_best_articles.json')

def calculate_doc_distance(doc1: {str: float}, doc2: {str: float}):
    total_common_frequency = 0
    for term, frequency in doc1.items():
        total_common_frequency += frequency * doc2.get(term, 0)
    return total_common_frequency


def term_frequency_inverse_document_frequency(tokenizer, article_collection):
    documents = []
    inverse_document_frequency = {}
    for article_list_no, article in tqdm(article_collection.items(), desc="Collecting terms"):
        for (producer, product_name, heading, hrefs, description) in article:
            term_frequency = {}
            total_tokens = 0
            for token in tokenizer(heading):
                incr(term_frequency, token.lemma_)
                total_tokens += 1
            for token in tokenizer(description):
                incr(term_frequency, token.lemma_)
                total_tokens += 1
            for token, number in term_frequency.items():
                incr(inverse_document_frequency, token)
                term_frequency[token] = number / total_tokens
            documents.append((producer, product_name, term_frequency))
    for token, doc_number in tqdm(inverse_document_frequency.items(), desc="computing inverses"):
        inverse_document_frequency[token] = math.log2(len(inverse_document_frequency) / doc_number)
    for producer, product_name, term_frequency in tqdm(documents, desc="computing td-idf"):
        for token, term_freq in term_frequency.items():
            term_frequency[token] = term_freq * inverse_document_frequency[token]
    return documents


def term_frequency_inverse_document_frequency_and_save():
    import spacy
    article_collection = load_json('my_best_articles.json')
    tokenizer = spacy.load("ja_core_news_sm")
    documents = term_frequency_inverse_document_frequency(tokenizer, article_collection)
    save_json(documents, 'my_best_articles_tf_idf.json')


def amazon_facets_with_prod_ids_to_facets_with_usage_counts(facets: {str: {str: [str]}}) -> {str: {str: int}}:
    return {facet: {val: len(prod_ids) for val, prod_ids in values.items()} for facet, values in facets.items()}


def amazon_extract_common_facets():
    facets = load_json('amazon_facets.json')
    facets = amazon_facets_with_prod_ids_to_facets_with_usage_counts(facets)

    save_json(facets, 'amazon_facets_counts.json')
