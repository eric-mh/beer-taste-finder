.PHONY: test
test: test_modeling test_preprocessing test_mongo test_pipeline

.PHONY: test_modeling test_preprocessing test_mongo test_pipeline
test_modeling:
	py.test tests/modeling_tests.py

test_preprocessing:
	py.test tests/preprocessor_tests.py

test_mongo:
	py.test tests/mongo_tests.py

test_pipeline:
	py.test tests/pipeline_tests.py
