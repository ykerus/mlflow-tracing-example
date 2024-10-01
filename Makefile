include .env

PYTHON ?= python

.PHONY: format
format:
	ruff check --fix
	ruff format
