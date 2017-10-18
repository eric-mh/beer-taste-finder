.PHONY: test
test:
	py.test tests/preprocessor_tests.py
	py.test tests/modeling_tests.py

.PHONY: test_modeling
test_modeling:
	py.test tests/modeling_tests.py

.PHONY: test_preprocessor
test_preprocessor:
	py.test tests/preprocessor_tests.py
