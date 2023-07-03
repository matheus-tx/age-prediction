.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:
SHELL := /bin/bash

install: 
	@echo "Installing poetry environment..."
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring && \
		curl -sSL https://install.python-poetry.org | python3 - -y && \
		poetry config --local virtualenvs.in-project true && \
		poetry init && \
		poetry install

	@echo "Initializing git repo..."
	git init

	@echo "Configuring poetry environment and git repo..."
	poetry env use python3.10
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring && \
		poetry add pre-commit && \
		poetry add mypy
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment..."
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring && \
		poetry env use python3.10

pull_data:
	poetry run dvc pull

test:
	pytest

docs_view:
	@echo View API documentation... 
	pdoc src --http localhost:8080

docs_save:
	@echo Save documentation to docs... 
	pdoc src -o docs

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache