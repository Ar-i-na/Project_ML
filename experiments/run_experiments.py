import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from config import RESULTS_DIR
from train import run_experiment

EXPERIMENTS = [
    dict(
        name="baseline_tfidf_raw",
        approach="tfidf_logreg",
        strip_meta=False,
        model_kwargs={"ngram_range": (1, 1), "C": 1.0},
        why=(
            "Базовое решение по заданию: TF-IDF (униграммы) + LogisticRegression "
            "на исходных (сырых) данных, без удаления headers/footers/quotes. "
        ),
    ),
    dict(
        name="tfidf_stripped_meta",
        approach="tfidf_logreg",
        strip_meta=True,
        model_kwargs={"ngram_range": (1, 1), "C": 1.0},
        why=(
            "Меняем strip_meta (True вместо False), "
            "всё остальное идентично baseline_tfidf_raw — так разница в metrics "
            "объясняется именно удалением служебных частей, а не чем-то ещё."
        ),
    ),
    dict(
        name="char_ngrams_raw",
        approach="tfidf_char_svm",
        strip_meta=False,
        model_kwargs={"ngram_range": (3, 5), "C": 1.0},
        why=(
            "Альтернативный подход: TF-IDF на символьных "
            "n-граммах (3-5 символов) вместо целых слов + LinearSVC. Сравниваем "
            "с baseline_tfidf_raw на тех же (сырых) данных, чтобы понять, "
            "помогает ли модели видеть подслова/морфологию вместо целых токенов."
        ),
    ),
    dict(
        name="char_ngrams_stripped_meta",
        approach="tfidf_char_svm",
        strip_meta=True,
        model_kwargs={"ngram_range": (3, 5), "C": 1.0},
        why=(
            "Проверяем, ведёт ли себя char n-gram подход так же, как обычный "
            "TF-IDF, при удалении headers/footers/quotes — то есть общий ли "
            "это эффект данных, или специфичный для конкретной репрезентации текста."
        ),
    ),
    dict(
        name="tfidf_stripped_bigrams",
        approach="tfidf_logreg",
        strip_meta=True,
        model_kwargs={"ngram_range": (1, 2), "C": 1.0},
        why=(
            "Доп. эксперимент: добавляем биграммы (например, 'space shuttle', "
            "'home run') поверх лучшей на данный момент конфигурации "
            "(tfidf_stripped_meta), чтобы проверить, помогает ли учитывать "
            "словосочетания, а не только отдельные слова."
        ),
    ),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []
    log_lines = ["# Лог экспериментов\n"]

    for exp in EXPERIMENTS:
        print(f"\n{'=' * 70}\nЗапускаю: {exp['name']}\n{'=' * 70}")
        result = run_experiment(
            approach=exp["approach"],
            strip_meta=exp["strip_meta"],
            model_kwargs=exp["model_kwargs"],
            save_prefix=exp["name"],
        )
        m = result["metrics"]
        summary_rows.append({
            "experiment": exp["name"],
            "approach": exp["approach"],
            "strip_meta": exp["strip_meta"],
            "model_kwargs": str(exp["model_kwargs"]),
            "macro_f1": round(m["macro_f1"], 4),
            "accuracy": round(m["accuracy"], 4),
        })

        log_lines.append(f"## {exp['name']}\n")
        log_lines.append(f"**Что изменили / конфигурация:** approach=`{exp['approach']}`, "
                          f"strip_meta=`{exp['strip_meta']}`, kwargs=`{exp['model_kwargs']}`\n")
        log_lines.append(f"**Зачем:** {exp['why']}\n")
        log_lines.append(f"**Результат:** macro-F1 = {m['macro_f1']:.4f}, "
                          f"accuracy = {m['accuracy']:.4f}\n")
        log_lines.append("**Вывод:** \n")

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_path = os.path.join(RESULTS_DIR, "experiments_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    log_lines.append("\n## Сводная таблица (отсортировано по macro-F1)\n")
    log_lines.append("```")
    log_lines.append(summary_df.to_string(index=False))
    log_lines.append("```")

    log_path = os.path.join(RESULTS_DIR, "experiment_log.md")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\n\nГотово. Сводная таблица: {summary_path}")
    print(f"Человекочитаемый лог: {log_path}")
    print("\n" + summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
