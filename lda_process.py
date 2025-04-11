import argparse
import os
import pickle
import string

from gensim import corpora
from gensim.models import LdaMulticore
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Define preprocessing function
def preprocess(doc):
    # Initialize stopwords and stemmer
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    # Tokenize, lowercase, and remove punctuation
    tokens = word_tokenize(doc.lower())
    tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in tokens]
    # Keep only alphabetic words and remove stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# Train LDA models with different number of topics
def train(num_topics_list, documents, dataset):
    texts = [preprocess(doc) for doc in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    for num_topics in num_topics_list:
        print(f"Training LDA model with {num_topics} topics...")
        lda_model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, workers=24, passes=15)

        output_dir = f'output/{dataset}/{num_topics}'
        os.makedirs(output_dir, exist_ok=True)

        lda_model.save(f'{output_dir}/lda.model')
        dictionary.save(f'{output_dir}/dictionary.dict')
        with open(f'{output_dir}/corpus.pkl', 'wb') as f:
            pickle.dump(corpus, f)
        print(f"Completed training for {num_topics} topics.")

# Generate topic distributions for documents
def get_topic_distribution(documents, dictionary, lda_model):
    distributions = []
    for doc in documents:
        bow = dictionary.doc2bow(preprocess(doc))
        topics = lda_model.get_document_topics(bow, minimum_probability=0)
        topic_distribution = [0] * lda_model.num_topics
        for topic_id, prob in topics:
            topic_distribution[topic_id] = prob
        # Normalize the distribution
        total = sum(topic_distribution)
        if total > 0:
            topic_distribution = [prob / total for prob in topic_distribution]
        distributions.append(topic_distribution)
    return distributions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'], help='Dataset name')
    args = parser.parse_args()
    dataset = args.dataset

    # Read all document contents
    with open(f'data/corpus/{dataset}_shuffle.txt', 'r') as f:
        doc_content_list = [line.strip() for line in f]

    num_topics_list = [20, 32, 64, 128, 256, 384, 512]
    train(num_topics_list=num_topics_list, documents=doc_content_list, dataset=dataset)
