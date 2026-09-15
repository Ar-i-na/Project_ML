import argparse
import json
import os
from sklearn.model_selection import train_test_split
from config import CATEGORIES, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from src.data import load_data
from src.evaluate import compute_metrics, plot_confusion_matrix, print_metrics
from src.models import build_pipeline
from src.preprocess import clean_corpus
def run_experiment(approach: str, strip_meta: bool, model_kwargs=None,
                    save_prefix=None, verbose=True):
    model_kwargs = model_kwargs or {}
    texts, labels, target_names = load_data(
        CATEGORIES, subset="all", strip_meta=strip_meta, random_state=RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=labels,
    )
    X_train = clean_corpus(X_train)
    X_test = clean_corpus(X_test)

    pipeline = build_pipeline(approach, **model_kwargs)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, target_names)
    if verbose:
        title = f"{approach} | strip_meta={strip_meta} | {model_kwargs}"
        print_metrics(metrics, title=title)

    cm = None
    if save_prefix:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        cm_path = os.path.join(RESULTS_DIR, f"{save_prefix}_confusion_matrix.png")
        cm = plot_confusion_matrix(
            y_test, y_pred, target_names, save_path=cm_path,
            title=f"{approach}, strip_meta={strip_meta}",
        )
        with open(os.path.join(RESULTS_DIR, f"{save_prefix}_metrics.json"), "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {
        "metrics": metrics,
        "cm": cm,
        "pipeline": pipeline,
        "target_names": target_names,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }

def parse_args():
    p = argparse.ArgumentParser(description="Обучить и оценить одну конфигурацию модели.")
    p.add_argument("--approach", default="tfidf_logreg",
                    choices=["tfidf_logreg", "tfidf_linearsvc", "w2v_logreg"])
    p.add_argument("--strip-meta", action="store_true",
                    help="Удалить headers/footers/quotes перед обучением.")
    p.add_argument("--ngram-max", type=int, default=1,
                    help="Верхняя граница n-грамм для tfidf_* подходов (1 = униграммы, 2 = + биграммы).")
    p.add_argument("--C", type=float, default=1.0, help="Параметр регуляризации классификатора.")
    p.add_argument("--save-prefix", default="baseline",
                    help="Префикс для файлов в results/ (metrics.json, confusion_matrix.png).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_kwargs = {"C": args.C}
    if args.approach in ("tfidf_logreg", "tfidf_linearsvc"):
        model_kwargs["ngram_range"] = (1, args.ngram_max)

    run_experiment(
        approach=args.approach,
        strip_meta=args.strip_meta,
        model_kwargs=model_kwargs,
        save_prefix=args.save_prefix,
    )
