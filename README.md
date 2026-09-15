# 20 Newsgroups Classification

Классификация текстов на 3 класса из датасета **20 Newsgroups**:
`comp.graphics`, `sci.space`, `rec.sport.baseball`.

**Исследовательский вопрос:** насколько служебные части сообщений
(headers, footers, quotes) влияют на качество классификации? Сравниваются
результаты на исходных данных и после удаления этих частей.

Подробное объяснение того, что делает каждый файл, и почему выбрана
метрика macro-F1, — в [`EXPLANATION.md`](./EXPLANATION.md).

## Установка

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Единственная зависимость нестандартной библиотеки — scikit-learn; никаких
компиляторов и системных библиотек не требуется, всё ставится через pipr
без проблем.

Датасет скачивается автоматически при первом запуске через
`sklearn.datasets.fetch_20newsgroups` (нужен интернет один раз, дальше
берётся из локального кэша `~/scikit_learn_data`).

## Структура проекта

```
.
├── config.py                    # классы, метрика, random_state — единая точка настройки
├── requirements.txt
├── train.py                     # запуск ОДНОГО эксперимента (CLI)
├── src/
│   ├── data.py                  # загрузка данных + EDA-обзор
│   ├── preprocess.py            # чистка текста
│   ├── models.py                # сборка sklearn Pipeline'ов (baseline + альтернатива)
│   ├── evaluate.py              # метрики + confusion matrix
│   └── error_analysis.py        # разбор ошибок, топ-фичи, confused pairs
├── experiments/
│   └── run_experiments.py       # запуск ВСЕХ экспериментов + сводная таблица
├── notebooks/
│   └── 01_eda.py                # первичный анализ данных (шаг 1 задания)
└── results/                     # сюда сохраняются метрики, confusion matrix, лог
```

## Альтернативный подход: TF-IDF на символьных n-граммах

Baseline использует TF-IDF на **словах** (каждый признак — целое слово).
Альтернатива (`tfidf_char_svm`) строит TF-IDF на **последовательностях
из 3–5 символов** (`analyzer="char_wb"` в `TfidfVectorizer`) + LinearSVC.
Модель видит не целые слова, а кусочки слов — суффиксы, устойчивые
сочетания букв, части терминов — что даёт принципиально другое
представление текста, устойчивое к редким словоформам и опечаткам.
Работает целиком на scikit-learn, без дополнительных зависимостей.

## Как запустить

**1. Первичный анализ данных (шаг 1 задания):**
```bash
python notebooks/01_eda.py
```

**2. Один эксперимент (например, baseline):**
```bash
python train.py --approach tfidf_logreg --save-prefix baseline
```
Удалить headers/footers/quotes: 
```bash
python train.py --approach tfidf_logreg --strip-meta --save-prefix stripped
```
Альтернативный подход (TF-IDF на символьных n-граммах + LinearSVC):
```bash
python train.py --approach tfidf_char_svm --save-prefix char_ngrams
```

**3. Все эксперименты сразу (рекомендуется):**
```bash
python experiments/run_experiments.py
```
Создаст:
- `results/experiments_summary.csv` — сводная таблица macro-F1/accuracy по всем запускам;
- `results/experiment_log.md` — человекочитаемый лог (что изменили / зачем / результат / вывод);
- `results/<experiment_name>_confusion_matrix.png` — confusion matrix для каждого эксперимента;
- `results/<experiment_name>_metrics.json` — детальные метрики по каждому классу.

**4. Анализ ошибок конкретной модели (в Python/ноутбуке):**
```python
from train import run_experiment
from src.error_analysis import get_misclassified, get_confused_pair, top_confused_pairs, top_features_per_class

result = run_experiment("tfidf_logreg", strip_meta=False, save_prefix="baseline")

# Примеры ошибочных предсказаний
get_misclassified(result["X_test"], result["y_test"], result["y_pred"], result["target_names"], n=10)

# Конкретная путаница: true=sci.space, predicted=comp.graphics
get_confused_pair(result["X_test"], result["y_test"], result["y_pred"], result["target_names"],
                   "sci.space", "comp.graphics")

# Топ-слова, на которые опирается модель по каждому классу (только для tfidf_* подходов)
top_features_per_class(result["pipeline"], result["target_names"])
```

## Готовые результаты

После запуска `experiments/run_experiments.py` результаты появятся в `results/`.
Файлы `experiment_log.md` и `experiments_summary.csv` содержат готовый
шаблон под фиксацию каждого эксперимента (что изменили / зачем / результат) —
вам останется дописать колонку "вывод" своими словами по итогам сравнения.

## Что уже заложено в код под ваш вопрос про headers/footers/quotes

- `src/data.load_data(..., strip_meta=True/False)` — единый флаг, который
  переключает `remove=('headers','footers','quotes')` в `fetch_20newsgroups`.
- `train.run_experiment` использует **одно и то же** train/test-разбиение
  (тот же `random_state`) для raw и strip_meta версий данных, чтобы разница
  в metrics была вызвана именно удалением служебных частей, а не другим
  случайным разбиением.
- В `experiments/run_experiments.py` уже есть пара экспериментов
  `baseline_tfidf_raw` vs `tfidf_stripped_meta`, различающихся только этим
  флагом — прямое сравнение для вашего исследовательского вопроса.
- `src/error_analysis.get_confused_pair` позволяет достать конкретные тексты
  для пары классов на raw и на stripped варианте отдельно — сравните их,
  чтобы найти примеры "ошибка исчезла/появилась после удаления headers".
