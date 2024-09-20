from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import spacy

class TextFeatureExtractor:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset_for_feature_extraction = dataset
        self.nlp = spacy.load('en_core_web_sm')
        self.dataset_features = {}

    def __filter_redundant_ngrams(self, ngrams, word_importance, additional_condition = False, sort = False):
        def __is_redundant(ngram, longer_ngram, importance = word_importance):
            # Преобразование n-gram'ов в множества слов
            ngram_words = set(ngram.split())
            longer_ngram_words = set(longer_ngram.split())

            # Проверка на пересечение множеств слов более чем на половину
            intersection = ngram_words.intersection(longer_ngram_words)
            if len(intersection) / len(longer_ngram_words) > 0.5:
                # For job and conditions
                if importance[ngram] <= importance[longer_ngram] and additional_condition:
                    return True
                return True
            return False
        
        # For 'why or why not'
        if sort:
            ngrams = sorted(ngrams, key = lambda x: len(x.split()), reverse=True)
        unique_ngrams = []
        for ngram in ngrams:
            # Проверяем, не является ли текущий n-gram дубликатом или подмножеством более длинного n-gram
            if not any(__is_redundant(ngram, longer_ngram) for longer_ngram in unique_ngrams):
                unique_ngrams.append(ngram)
        return unique_ngrams

    def __preprocess_text(self, text):
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    def __check_ngrams_in_text(self, text, ngrams, word_importance):
        # Препроцессинг текста для получения лемм
        tokens = self.__preprocess_text(text)
        token_set = tokens.split()

        def ngram_in_tokens(ngram):
            ngram_tokens = ngram.split()  # Разбиваем n-грамму на отдельные слова
            # Проверяем, содержится ли каждая часть n-граммы в токенах текста
            return all(token in token_set for token in ngram_tokens)

        # Отбираем только те n-граммы, которые присутствуют в тексте
        matching_ngrams = [ngram for ngram in ngrams if ngram_in_tokens(ngram)]

        if not matching_ngrams or len(token_set) < 2:
            most_important_ngram = max(ngrams, key=lambda ngram: word_importance[ngram])
            return most_important_ngram

        most_important_ngram = max(matching_ngrams, key=lambda ngram: word_importance[ngram])
        return most_important_ngram
    
    def __define_frequent_words_in_column(self, dataset: pd.DataFrame, column: str):
        diagnose_and_work_columns = ["If yes, what condition(s) have you been diagnosed with?",
                    "If so, what condition(s) were you diagnosed with?",
                    "Which of the following best describes your work position?"]
        if column in diagnose_and_work_columns:
            count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 3), binary=False, token_pattern = r'\b\w[\w\'-]*\b')

            x_counts = count_vectorizer.fit_transform(dataset[column])
            feature_names = count_vectorizer.get_feature_names_out()

            ngram_features = pd.DataFrame(x_counts.toarray(), columns=feature_names)
            word_importance = ngram_features.sum().sort_values(ascending=False)
            filtered_word_importance = self.__filter_redundant_ngrams(word_importance.index, word_importance, additional_condition=True)
        else:
            dataset[column] = dataset[column].apply(self.__preprocess_text)
            count_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 3), binary=False, token_pattern = r'\b\w[\w\'-]*\b')

            x_counts = count_vectorizer.fit_transform(dataset[column])
            feature_names = count_vectorizer.get_feature_names_out()
            tfidf_transformer = TfidfTransformer()
            x_tfidf = tfidf_transformer.fit_transform(x_counts)

            tfidf_features = pd.DataFrame(x_tfidf.toarray(), columns=feature_names)
            word_importance = tfidf_features.sum().sort_values(ascending=False)
            filtered_word_importance = self.__filter_redundant_ngrams(word_importance.index, word_importance, sort = True)

        word_importance = word_importance.loc[filtered_word_importance]
        word_importance = word_importance.sort_values(ascending=False)
        # top_n_words = word_importance.head(20)

        # TODO
        ngrams = word_importance.index.tolist()
        dataset[column] = dataset[column].\
            apply(lambda x: self.__check_ngrams_in_text(x, ngrams, word_importance))


    def extract_features(self):
        for ith_column in self.dataset_for_feature_extraction.columns:
            self.__define_frequent_words_in_column(self.dataset_for_feature_extraction, ith_column)
