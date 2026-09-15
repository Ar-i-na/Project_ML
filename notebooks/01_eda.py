import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CATEGORIES
from src.data import dataset_overview, load_data
texts_raw, labels, target_names = load_data(CATEGORIES, subset="all", strip_meta=False)

print(f"\nВсего документов: {len(texts_raw)}")
print("\nРаспределение по классам:")
print(dataset_overview(texts_raw, labels, target_names))

print("\nПример документа (raw)")
print(texts_raw[0][:600])

print("\nДанные без headers/footers/quotes")
texts_stripped, _, _ = load_data(CATEGORIES, subset="all", strip_meta=True)
print(texts_stripped[0][:600])

print("\nСравнение средней длины текста (в словах) raw vs stripped:")
raw_lengths = [len(t.split()) for t in texts_raw]
stripped_lengths = [len(t.split()) for t in texts_stripped]
print(f"raw:      среднее = {sum(raw_lengths)/len(raw_lengths):.1f} слов")
print(f"stripped: среднее = {sum(stripped_lengths)/len(stripped_lengths):.1f} слов")
