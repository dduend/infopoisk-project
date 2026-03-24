import argparse
import os
import zipfile
import pandas as pd
from preprocessing import preprocess_text, preprocess_query
from indices import BM25SIndex, Word2VecIndex, FastTextIndex


DATA_PATH = r'C:\Users\Далия\PycharmProjects\kafka_search\data\films_data.csv.zip'


def load_corpus():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'файл не найден: {DATA_PATH}')

    extract_dir = os.path.join(os.path.dirname(DATA_PATH), 'movie_plots')

    with zipfile.ZipFile(DATA_PATH, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    csv_path = os.path.join(extract_dir, 'films_data.csv')
    df = pd.read_csv(csv_path)

    print(f'исходный размер: {df.shape}')

    # убираем пустые
    df = df[df['plot'].notna()].copy()
    print(f'после удаления nan: {df.shape}')

    # берем первые 2000
    df_sample = df.head(2000).copy()

    # предобработка
    results = [preprocess_text(text) for text in df_sample['plot']]
    processed_texts = [r[1] for r in results]

    # убираем пустые
    mask = [len(t) > 0 for t in processed_texts]
    df_sample = df_sample[mask].reset_index(drop=True)
    processed_texts = [t for t in processed_texts if len(t) > 0]

    original_texts = df_sample['plot'].tolist()

    print(f'загружено {len(original_texts)} документов')

    return original_texts, processed_texts


def main():
    parser = argparse.ArgumentParser(
        description='поиск по сюжетам фильмов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
примеры:
  python main.py -q "история любви"
  python main.py -q "детектив убийство" -i bm25 -k 10
  python main.py -q "смешная комедия" -i word2vec -k 5
  python main.py -q "приключения в космосе" -i fasttext -k 3
        '''
    )

    parser.add_argument('-q', '--query', type=str, required=True, help='поисковый запрос')
    parser.add_argument('-i', '--index', type=str, choices=['bm25', 'word2vec', 'fasttext'],
                        default='bm25', help='тип индекса (по умолчанию bm25)')
    parser.add_argument('-k', '--top-k', type=int, default=5, help='количество результатов')

    args = parser.parse_args()

    print(f'запрос: {args.query}')
    print(f'индекс: {args.index}')
    print(f'top-{args.top_k}')

    try:
        original_texts, processed_texts = load_corpus()
    except Exception as e:
        print(e)
        return

    if args.index == 'bm25':
        index = BM25SIndex(k1=1.5, b=0.75)
    elif args.index == 'word2vec':
        index = Word2VecIndex(vector_size=100)
    else:
        index = FastTextIndex(vector_size=100, epoch=10)

    index.build(original_texts, processed_texts)

    results = index.search(args.query, top_k=args.top_k)

    if len(results) > 0:
        for i, row in results.iterrows():
            print(f'\n{i+1}) оценка: {row["score"]:.4f}')
            preview = row['text'][:300]
            if len(row['text']) > 300:
                preview = preview + '...'
            print(f'{preview}')
    else:
        print('\nничего не найдено')


if __name__ == '__main__':
    main()
