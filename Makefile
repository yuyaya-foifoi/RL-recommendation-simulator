SRC := src

style:
	pip install -r ./requirements/style.txt
	isort $(SRC)
	black $(SRC) --line-length 79
	flake8 $(SRC)

setup:
	pip install -r ./requirements/library.txt