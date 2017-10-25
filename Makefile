.PHONY: test
test: test_modeling test_preprocessing test_mongo test_pipeline

.PHONY: test_modeling test_preprocessing test_mongo test_pipeline
test_modeling:
	python -m pytest tests/modeling_tests.py

test_preprocessing:
	python -m pytest tests/preprocessor_tests.py

test_mongo:
	python -m pytest tests/mongo_tests.py

test_pipeline:
	python -m pytest tests/pipeline_tests.py
