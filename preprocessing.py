import re
import nltk
import pymorphy3
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

morph = pymorphy3.MorphAnalyzer()

russian_stopwords = set(nltk_stopwords.words('russian'))

# специфические для кино слова
movie_stopwords = {
    'фильм', 'фильма', 'фильме', 'фильмы', 'кино', 'картина',
    'лента', 'режиссер', 'режиссера', 'сценарий', 'актер', 'актера',
    'актеры', 'роль', 'роли', 'играет', 'сыграл', 'снимался', 'снял',
    'сериал', 'эпизод', 'серия', 'кадр', 'сцена', 'диалог', 'монолог'
}
stopwords = russian_stopwords.union(movie_stopwords)


def preprocess_text(text):
    if not isinstance(text, str):
        return [], ''

    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z\s]', ' ', text)
    text = text.lower()

    words = text.split()

    # лемматизация и удаление стоп-слов
    tokens = []
    for word in words:
        if word not in stopwords and len(word) > 2:
            lemma = morph.parse(word)[0].normal_form
            tokens.append(lemma)

    return tokens, ' '.join(tokens)


def preprocess_query(query):
    if not isinstance(query, str):
        return ''

    query = re.sub(r'[^\w\s]', '', query)
    query = query.lower()
    tokens = word_tokenize(query)
    tokens = [morph.parse(t)[0].normal_form for t in tokens]

    return ' '.join(tokens)
