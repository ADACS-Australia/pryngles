##################################################################
#                                                                #
#.#####...#####...##..##..##..##...####...##......######...####..#
#.##..##..##..##...####...###.##..##......##......##......##.....#
#.#####...#####.....##....##.###..##.###..##......####.....####..#
#.##......##..##....##....##..##..##..##..##......##..........##.#
#.##......##..##....##....##..##...####...######..######...####..#
#................................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################

.PHONY: release show docs docs-clean

# Prefer local venv python if present
PYTHON ?= python3
ifeq ($(wildcard .venv/bin/python),.venv/bin/python)
PYTHON := .venv/bin/python
endif

##################################################################
#BASIC RULES
##################################################################
clean:cleancrap

cleanall:cleancrap cleanout cleandist

#=========================
#Clean
#=========================
cleancrap:
	@echo "Cleaning crap..."
	@-find . -name "*~" -delete
	@-find . -name "#*#" -delete
	@-find . -name "#*" -delete
	@-find . -name ".#*" -delete
	@-find . -name ".#*#" -delete
	@-find . -name ".DS_Store" -delete
	@-find . -name "Icon*" -delete
	@-find . -name "*.egg-info*" -type d | xargs rm -fr

cleanout:
	@echo "Cleaning all compiled objects..."
	@-find . -name "*.o" -delete
	@-find . -name "*.opp" -delete
	@-find . -name "*.gcno" -delete
	@-find . -name "*.gcda" -delete
	@-find . -name "*.gcov" -delete
	@-find . -name "*.info" -delete
	@-find . -name "*.out" -delete
	@-find . -name "*.tout" -delete
	@-find . -name "*.so" -delete
	@-find . -name '__pycache__' -type d | xargs rm -fr
	@-find . -name '.ipynb_checkpoints' -type d | xargs rm -fr

cleandist:
	@-rm -rf dist/
	@-rm -rf build/

##################################################################
#TEST
##################################################################
test:
	@$(PYTHON) -m pytest

##################################################################
#GIT
##################################################################
commit:
	@echo "Commiting..."
	@-git commit -am "Commit"
	@-git push

pull:
	@echo "Pulling new files..."
	@-git pull 

##################################################################
#PACKAGE RULES
##################################################################
# Example:
#   make release RELMODE=release VERSION=1.0.0
#   make release RELMODE=test VERSION=1.0.0
RELMODE ?= release
VERSION ?=

show:
	@python3 -c "import re, pathlib; t=pathlib.Path('setup.py').read_text(encoding='utf-8'); print('setup.py:', re.search(r\"^\\s*version\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]\", t, re.M).group(1))"
	@python3 -c "from pryngles.version import version; print('package :', version)"

release:
	@echo "Releasing a new version..."
	@bash bin/release.sh $(RELMODE) $(VERSION)

pipinstall:
	@$(PIP) install -e .

import:
	@python3 -c "from pryngles import *;print(version)"

##################################################################
#REQUIREMENTS
##################################################################
requirements:
	@python3 bin/generate_requirements.py
	@python3 -m pip install -r requirements.txt

##################################################################
# DOCS
##################################################################
docs:
	@$(PYTHON) -m pip install -U pip
	@$(PYTHON) -m pip install -e .
	@$(PYTHON) -m pip install -r docs/requirements.txt
	@rm -rf docs/_build
	@$(PYTHON) -m sphinx.cmd.build -M html docs docs/_build

docs-clean:
	@rm -rf docs/_build
