# Дополнительные исследовательские скрипты

Исторические скрипты из рабочей папки проекта. Сохранены для прозрачности экспериментов; не являются гарантированно воспроизводимым pipeline. Запуск из корня репозитория.

- `extract_base_facts.py`: извлечение фактов Multi-Session Chat (выход `base_facts.jsonl`, не полный оригинальный `base_facts_upd.jsonl`).
- `generate_examples.py`, `balance_dataset.py`, `filter_dataset.py`, `filter_cross_class_duplicates.py`, `refresh_delete.py`: генерация, очистка и балансировка. Требуются отсутствующие промежуточные данные и исходные шаблоны `prompts/prompt_one_tag2.txt`, `prompts/delete_2.txt`.
- `analyze_lexical_diversity.py`, `sample_manual_validation.py`: вспомогательная оценка разнообразия и выборка для разметки.
- `build_tfidf_context.py`, `build_multifact_training_data.py`, `train_multifact_roberta.py`, `evaluate_qwen_multifact.py`, `evaluate_ood.py`, `plot_confusion_matrix.py`, `inspect_multifact.py`: дополнительные варианты работы с контекстом из нескольких фактов. Требуются оригинальные данные/модели. Названия промежуточных файлов отличаются между версиями; эти версии не образуют готовую цепочку запуска. Текущая версия `evaluate_qwen_multifact.py` запрашивает только операцию, а не ID факта; её нельзя считать подтверждением опубликованной метрики выбора целевого факта.

Не включены пустой файл, локальная проверка API-ключа и три пилотных варианта генерации. Оригинальные файлы сохранены у автора в рабочем архиве.
