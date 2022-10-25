export PATH := /root/.local/bin:$(PATH)

install:
	apt-get update
	apt-get install python3.9
	apt-get install python3.9-distutils
	curl -sSL https://install.python-poetry.org | python3 - --version 1.1.14
	poetry install

train:
	poetry run python train.py
