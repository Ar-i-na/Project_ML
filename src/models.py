from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def build_pipeline(approach: str, **kwargs) -> Pipeline:
    if approach == "tfidf_logreg":
        vect_kwargs = dict(
            ngram_range=kwargs.get("ngram_range", (1, 1)),
            min_df=kwargs.get("min_df", 2),
            max_features=kwargs.get("max_features", 20000),
            sublinear_tf=True,
        )
        return Pipeline([
            ("tfidf", TfidfVectorizer(**vect_kwargs)),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=kwargs.get("C", 1.0),
                random_state=42,
            )),
        ])

    if approach == "tfidf_linearsvc":
        vect_kwargs = dict(
            ngram_range=kwargs.get("ngram_range", (1, 1)),
            min_df=kwargs.get("min_df", 2),
            max_features=kwargs.get("max_features", 20000),
            sublinear_tf=True,
        )
        return Pipeline([
            ("tfidf", TfidfVectorizer(**vect_kwargs)),
            ("clf", LinearSVC(
                C=kwargs.get("C", 1.0),
                random_state=42,
            )),
        ])

    if approach == "tfidf_char_svm":
        vect_kwargs = dict(
            analyzer="char_wb",  # n-граммы символов внутри границ слов
            ngram_range=kwargs.get("ngram_range", (3, 5)),
            min_df=kwargs.get("min_df", 2),
            max_features=kwargs.get("max_features", 30000),
            sublinear_tf=True,
        )
        return Pipeline([
            ("tfidf", TfidfVectorizer(**vect_kwargs)),
            ("clf", LinearSVC(
                C=kwargs.get("C", 1.0),
                random_state=42,
            )),
        ])

    raise ValueError(f"Неизвестный approach: {approach}")

