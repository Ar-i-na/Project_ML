from collections import Counter
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

def load_data(categories, subset="train", strip_meta=False, random_state=67):
    remove = ("headers", "footers", "quotes") if strip_meta else ()

    bunch = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=remove,
        random_state=random_state,
        shuffle=True,
    )
    return bunch.data, bunch.target, bunch.target_names

def dataset_overview(texts, labels, target_names):
    counts = Counter(labels)
    rows = []
    for idx, name in enumerate(target_names):
        class_texts = [t for t, l in zip(texts, labels) if l == idx]
        lengths = [len(t.split()) for t in class_texts]
        rows.append({
            "class": name,
            "n_docs": counts[idx],
            "avg_words": sum(lengths) / len(lengths) if lengths else 0,
            "min_words": min(lengths) if lengths else 0,
            "max_words": max(lengths) if lengths else 0,
        })
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from config import CATEGORIES

    texts, labels, names = load_data(CATEGORIES, subset="all", strip_meta=False)
    print(f"Всего документов: {len(texts)}")
    print(dataset_overview(texts, labels, names))

    print("\nПример документа (raw, с headers/footers/quotes):")
    print(texts[0][:500])

    texts_clean, _, _ = load_data(CATEGORIES, subset="all", strip_meta=True)
    print("\nТот же тип документа, но strip_meta=True:")
    print(texts_clean[0][:500])
