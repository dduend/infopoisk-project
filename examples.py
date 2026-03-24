import time
from main import load_corpus
from indices import BM25SIndex, Word2VecIndex, FastTextIndex


def run_examples():
    try:
        original_texts, processed_texts = load_corpus()
    except Exception as e:
        print(e)
        return

    # тестовые запросы
    test_queries = [
        {
            'query': 'история любви',
            'index': 'bm25',
            'description': 'поиск фильмов о романтических отношениях',
            'expected': 'фильмы с ключевыми словами "история", "любовь"'
        },
        {
            'query': 'детектив убийство',
            'index': 'bm25',
            'description': 'поиск детективных фильмов с убийствами',
            'expected': 'фильмы с ключевыми словами "детектив", "убийство"'
        },
        {
            'query': 'смешная комедия',
            'index': 'word2vec',
            'description': 'поиск комедий по смыслу (не только по точным словам)',
            'expected': 'фильмы со словами "смешной", "комедия" и похожими по смыслу'
        },
        {
            'query': 'приключения в космосе',
            'index': 'fasttext',
            'description': 'семантический поиск научно-фантастических фильмов',
            'expected': 'фильмы о космосе, приключениях, фантастике'
        },
        {
            'query': 'романтическая драма',
            'index': 'word2vec',
            'description': 'поиск фильмов по смысловому сходству',
            'expected': 'фильмы с похожей тематикой'
        },
        {
            'query': 'военный фильм',
            'index': 'bm25',
            'description': 'поиск военных фильмов по ключевым словам',
            'expected': 'фильмы со словами "война", "военный"'
        },
        {
            'query': 'фантастика будущее',
            'index': 'fasttext',
            'description': 'семантический поиск научной фантастики',
            'expected': 'фильмы о будущем, технологиях, фантастике'
        },
        {
            'query': 'триллер страх',
            'index': 'bm25',
            'description': 'поиск триллеров и фильмов ужасов',
            'expected': 'фильмы с ключевыми словами "триллер", "страх"'
        }
    ]

    for i, test in enumerate(test_queries, 1):
        print(f'пример {i}: {test["query"]}')
        print(f'тип индекса: {test["index"]}')
        print(f'описание: {test["description"]}')
        print(f'ожидаемые результаты: {test["expected"]}')

        # выбираем индекс
        if test['index'] == 'bm25':
            index = BM25SIndex(k1=1.5, b=0.75)
        elif test['index'] == 'word2vec':
            index = Word2VecIndex(vector_size=100)
        else:
            index = FastTextIndex(vector_size=100, epoch=10)

        build_start = time.time()
        index.build(original_texts, processed_texts)
        build_time = time.time() - build_start
        print(f'время построения индекса: {build_time:.2f} сек')

        results = index.search(test['query'], top_k=3)

        if len(results) > 0:
            print(f'\nтоп-{len(results)} результатов:')
            for j, row in results.iterrows():
                print(f'\n {j+1}) оценка: {row["score"]:.4f}')
                preview = row['text'][:200]
                if len(row['text']) > 200:
                    preview = preview + '...'
                print(f'{preview}')
        else:
            print('\nничего не найдено')


def run_quick_demo():
    try:
        original_texts, processed_texts = load_corpus()
    except Exception as e:
        print(e)
        return

    query = 'история любви'

    # индексы для демонстрации
    indices = [
        ('BM-25', BM25SIndex()),
        ('Word2Vec', Word2VecIndex(vector_size=100)),
        ('FastText', FastTextIndex(vector_size=100, epoch=10))
    ]

    for name, index in indices:
        print(f'\nиндекс: {name}')
        print(f'запрос: {query}')

        build_start = time.time()
        index.build(original_texts, processed_texts)
        build_time = time.time() - build_start

        print(f'время построения: {build_time:.2f} сек')

        results = index.search(query, top_k=3)

        if len(results) > 0:
            print(f'\nтоп-{len(results)} результатов:')
            for j, row in results.iterrows():
                print(f'\n{j+1}) оценка: {row["score"]:.4f}')
                preview = row['text'][:200]
                if len(row['text']) > 200:
                    preview = preview + '...'
                print(f'{preview}')
        else:
            print('\nничего не найдено')


if __name__ == '__main__':
    print('\nвыберите режим демонстрации:')
    print('1 полная демонстрация (8 примеров)')
    print('2 быстрая демонстрация (3 индекса, 1 запрос)')

    choice = input('\nваш выбор (1-2): ').strip()

    if choice == '1':
        run_examples()
    elif choice == '2':
        run_quick_demo()
