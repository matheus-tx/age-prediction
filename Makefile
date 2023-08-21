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

	@echo "Configuring poetry environment..."
	poetry env use python3.10

configure:
	backends.null.Keyring && \
		poetry add pre-commit && \
		poetry add mypy
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment..."
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring && \
		poetry env use python3.10
