import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import bm25s
from gensim.models import Word2Vec
from preprocessing import preprocess_query
import tempfile
import os
import fasttext


class BM25SIndex:
    def __init__(self, k1=1.5, b=0.75, method='lucene'):
        self.k1 = k1
        self.b = b
        self.method = method
        self.retriever = bm25s.BM25(method=method, k1=k1, b=b)
        self.corpus = []
        self.original_texts = []
        self.processed_texts = []
        self.is_built = False

    def build(self, original_texts, processed_texts):
        self.original_texts = original_texts
        self.processed_texts = processed_texts
        self.corpus = processed_texts

        corpus_tokens = [text.split() for text in processed_texts]

        self.retriever.index(corpus_tokens)
        self.is_built = True

    def search(self, query, top_k=5):
        if not self.is_built:
            raise ValueError('индекс не построен')

        start_time = time.time()

        processed_query = preprocess_query(query)
        query_tokens = processed_query.split() if processed_query else []

        if not query_tokens:
            print('запрос пустой после обработки')
            return pd.DataFrame()

        # оборачиваем query_tokens в список
        results, scores = self.retriever.retrieve([query_tokens], k=top_k)

        search_time = time.time() - start_time
        print(f'время поиска: {search_time:.4f} сек')

        results_list = []
        if len(results) > 0 and len(results[0]) > 0:
            for i, (doc_idx, score) in enumerate(zip(results[0], scores[0])):
                text = self.original_texts[doc_idx]
                if len(text) > 500:
                    text = text[:500] + '...'

                results_list.append({
                    'index': doc_idx,
                    'text': text,
                    'score': score
                })

        return pd.DataFrame(results_list)


class Word2VecIndex:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.doc_vectors = None
        self.original_texts = []
        self.processed_texts = []
        self.is_built = False

    def build(self, original_texts, processed_texts):
        self.original_texts = original_texts
        self.processed_texts = processed_texts

        tokenized_docs = [text.split() for text in processed_texts]

        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=0
        )

        # построение векторов документов
        self.doc_vectors = []
        words_found = 0
        words_not_found = 0

        for doc_tokens in tokenized_docs:
            word_vectors = []
            for token in doc_tokens:
                if token in self.model.wv:
                    word_vectors.append(self.model.wv[token])
                    words_found = words_found + 1
                else:
                    words_not_found = words_not_found + 1

            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)

            self.doc_vectors.append(doc_vector)

        self.doc_vectors = np.array(self.doc_vectors)
        self.is_built = True

    def search(self, query, top_k=5):
        if not self.is_built:
            raise ValueError('индекс не построен')

        processed_query = preprocess_query(query)
        query_tokens = processed_query.split()

        if not query_tokens:
            print('запрос пустой после обработки')
            return pd.DataFrame()

        start_time = time.time()

        # построение вектора запроса
        query_vectors = []
        for token in query_tokens:
            if token in self.model.wv:
                query_vectors.append(self.model.wv[token])

        if query_vectors:
            query_vector = np.mean(query_vectors, axis=0)
        else:
            print('слова запроса не найдены в модели')
            query_vector = np.zeros(self.vector_size)

        similarities = cosine_similarity([query_vector], self.doc_vectors).flatten()

        search_time = time.time() - start_time
        print(f'время поиска: {search_time:.4f} сек')

        # нормализуем оценки
        if np.max(similarities) > 0:
            similarities = similarities / np.max(similarities)

        indices_with_scores = [(i, sim) for i, sim in enumerate(similarities) if sim > 0]
        indices_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in indices_with_scores[:top_k]]

        results = []
        for idx in top_indices:
            text = self.original_texts[idx]
            if len(text) > 500:
                text = text[:500] + '...'

            results.append({
                'index': idx,
                'text': text,
                'score': similarities[idx]
            })

        return pd.DataFrame(results)


class FastTextIndex:
    def __init__(self, vector_size=100, window=5, min_count=1, epoch=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epoch = epoch
        self.model = None
        self.doc_vectors = None
        self.original_texts = []
        self.processed_texts = []
        self.is_built = False

    def build(self, original_texts, processed_texts):
        self.original_texts = original_texts
        self.processed_texts = processed_texts

        # записываем корпус во временный файл
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            for text in processed_texts:
                f.write(text + '\n')
            temp_file = f.name

        try:
            # обучаем модель на корпусе
            self.model = fasttext.train_unsupervised(
                temp_file,
                model='skipgram',
                dim=self.vector_size,
                ws=self.window,
                minCount=self.min_count,
                epoch=self.epoch,
                thread=4,
                verbose=1
            )

        except Exception as e:
            print(e)
            self.is_built = False
            return
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        tokenized_docs = [text.split() for text in processed_texts]

        self.doc_vectors = []
        words_found = 0
        words_not_found = 0

        for doc_tokens in tokenized_docs:
            word_vectors = []
            for token in doc_tokens:
                try:
                    vec = self.model.get_word_vector(token)
                    if vec is not None and not np.all(vec == 0):
                        word_vectors.append(vec)
                        words_found = words_found + 1
                    else:
                        words_not_found = words_not_found + 1
                except:
                    words_not_found = words_not_found + 1

            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)

            self.doc_vectors.append(doc_vector)

        self.doc_vectors = np.array(self.doc_vectors)
        self.is_built = True

    def search(self, query, top_k=5):
        if not self.is_built:
            raise ValueError('индекс не построен')

        start_time = time.time()

        processed_query = preprocess_query(query)
        query_tokens = processed_query.split()

        if not query_tokens:
            print('запрос пустой после обработки')
            return pd.DataFrame()

        query_vectors = []
        for token in query_tokens:
            try:
                vec = self.model.get_word_vector(token)
                if vec is not None and not np.all(vec == 0):
                    query_vectors.append(vec)
            except:
                pass

        if query_vectors:
            query_vector = np.mean(query_vectors, axis=0)
        else:
            print('слова запроса не найдены в модели')
            query_vector = np.zeros(self.vector_size)

        similarities = cosine_similarity([query_vector], self.doc_vectors).flatten()

        search_time = time.time() - start_time
        print(f'время поиска: {search_time:.4f} сек')

        if np.max(similarities) > 0:
            similarities = similarities / np.max(similarities)

        indices_with_scores = [(i, sim) for i, sim in enumerate(similarities) if sim > 0]
        indices_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in indices_with_scores[:top_k]]

        results = []
        for idx in top_indices:
            text = self.original_texts[idx]
            if len(text) > 500:
                text = text[:500] + '...'

            results.append({
                'index': idx,
                'text': text,
                'score': similarities[idx]
            })

        return pd.DataFrame(results)
