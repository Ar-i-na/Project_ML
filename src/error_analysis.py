import pandas as pd
def get_misclassified(texts, y_true, y_pred, target_names, n=10):
    rows = []
    for text, yt, yp in zip(texts, y_true, y_pred):
        if yt != yp:
            rows.append({
                "true": target_names[yt],
                "predicted": target_names[yp],
                "text_snippet": text[:300].replace("\n", " "),
            })
    df = pd.DataFrame(rows)
    return df.head(n) if n else df

def get_confused_pair(texts, y_true, y_pred, target_names, class_a, class_b, n=5):
    df = get_misclassified(texts, y_true, y_pred, target_names, n=None)
    subset = df[(df["true"] == class_a) & (df["predicted"] == class_b)]
    return subset.head(n)


def top_confused_pairs(cm, target_names, top_k=5):
    pairs = []
    for i, true_name in enumerate(target_names):
        for j, pred_name in enumerate(target_names):
            if i != j:
                pairs.append((true_name, pred_name, cm[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def top_features_per_class(pipeline, target_names, top_n=15):
    vectorizer = pipeline.named_steps.get("tfidf")
    clf = pipeline.named_steps.get("clf")
    if vectorizer is None or not hasattr(clf, "coef_"):
        raise ValueError(
            "top_features_per_class работает только для tfidf_* pipeline "
            "с линейным классификатором, имеющим coef_."
        )
    feature_names = vectorizer.get_feature_names_out()
    result = {}
    for idx, class_name in enumerate(target_names):
        coefs = clf.coef_[idx]
        top_idx = coefs.argsort()[-top_n:][::-1]
        result[class_name] = [(feature_names[i], round(coefs[i], 3)) for i in top_idx]
    return result
