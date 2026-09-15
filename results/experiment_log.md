# Лог экспериментов

## baseline_tfidf_raw

**Что изменили / конфигурация:** approach=`tfidf_logreg`, strip_meta=`False`, kwargs=`{'ngram_range': (1, 1), 'C': 1.0}`

**Зачем:** Базовое решение по заданию: TF-IDF (униграммы) + LogisticRegression на исходных (сырых) данных, без удаления headers/footers/quotes. Это отправная точка для сравнения со всеми остальными вариантами.

**Результат:** macro-F1 = 0.9704, accuracy = 0.9702

**Вывод:** _(заполните после сравнения с другими экспериментами — см. summary таблицу ниже)_

## tfidf_stripped_meta

**Что изменили / конфигурация:** approach=`tfidf_logreg`, strip_meta=`True`, kwargs=`{'ngram_range': (1, 1), 'C': 1.0}`

**Зачем:** ОСНОВНОЙ исследовательский вопрос: насколько headers/footers/quotes влияют на качество? Меняем только strip_meta (True вместо False), всё остальное идентично baseline_tfidf_raw — так разница в metrics объясняется именно удалением служебных частей, а не чем-то ещё.

**Результат:** macro-F1 = 0.8944, accuracy = 0.8945

**Вывод:** _(заполните после сравнения с другими экспериментами — см. summary таблицу ниже)_

## char_ngrams_raw

**Что изменили / конфигурация:** approach=`tfidf_char_svm`, strip_meta=`False`, kwargs=`{'ngram_range': (3, 5), 'C': 1.0}`

**Зачем:** Альтернативный подход (шаг 4 задания): TF-IDF на символьных n-граммах (3-5 символов) вместо целых слов + LinearSVC. Сравниваем с baseline_tfidf_raw на тех же (сырых) данных, чтобы понять, помогает ли модели видеть подслова/морфологию вместо целых токенов.

**Результат:** macro-F1 = 0.9797, accuracy = 0.9797

**Вывод:** _(заполните после сравнения с другими экспериментами — см. summary таблицу ниже)_

## char_ngrams_stripped_meta

**Что изменили / конфигурация:** approach=`tfidf_char_svm`, strip_meta=`True`, kwargs=`{'ngram_range': (3, 5), 'C': 1.0}`

**Зачем:** Проверяем, ведёт ли себя char n-gram подход так же, как обычный TF-IDF, при удалении headers/footers/quotes — то есть общий ли это эффект данных, или специфичный для конкретной репрезентации текста.

**Результат:** macro-F1 = 0.9065, accuracy = 0.9066

**Вывод:** _(заполните после сравнения с другими экспериментами — см. summary таблицу ниже)_

## tfidf_stripped_bigrams

**Что изменили / конфигурация:** approach=`tfidf_logreg`, strip_meta=`True`, kwargs=`{'ngram_range': (1, 2), 'C': 1.0}`

**Зачем:** Доп. эксперимент: добавляем биграммы (например, 'space shuttle', 'home run') поверх лучшей на данный момент конфигурации (tfidf_stripped_meta), чтобы проверить, помогает ли учитывать словосочетания, а не только отдельные слова.

**Результат:** macro-F1 = 0.8835, accuracy = 0.8836

**Вывод:** _(заполните после сравнения с другими экспериментами — см. summary таблицу ниже)_


## Сводная таблица (отсортировано по macro-F1)

```
               experiment       approach  strip_meta                      model_kwargs  macro_f1  accuracy
          char_ngrams_raw tfidf_char_svm       False {'ngram_range': (3, 5), 'C': 1.0}    0.9797    0.9797
       baseline_tfidf_raw   tfidf_logreg       False {'ngram_range': (1, 1), 'C': 1.0}    0.9704    0.9702
char_ngrams_stripped_meta tfidf_char_svm        True {'ngram_range': (3, 5), 'C': 1.0}    0.9065    0.9066
      tfidf_stripped_meta   tfidf_logreg        True {'ngram_range': (1, 1), 'C': 1.0}    0.8944    0.8945
   tfidf_stripped_bigrams   tfidf_logreg        True {'ngram_range': (1, 2), 'C': 1.0}    0.8835    0.8836
```