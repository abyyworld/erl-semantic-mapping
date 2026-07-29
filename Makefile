.DEFAULT_GOAL := help
PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin
SCENE ?= conf/scenes/room_v1.yaml
IMAGE ?= erl-semantic-mapping

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install with dev tools
	$(BIN)/pip install -e '.[dev]'

.PHONY: install-locked
install-locked: $(BIN)/python ## Install the exact hash-pinned environment CI and Docker use
	$(BIN)/pip install --require-hashes --no-deps -r requirements-dev.lock
	$(BIN)/pip install --no-deps -e .

.PHONY: lock
lock: ## Regenerate requirements.lock and requirements-dev.lock from pyproject.toml
	$(BIN)/pip install pip-tools
	$(BIN)/pip-compile --generate-hashes --strip-extras -o requirements.lock pyproject.toml
	$(BIN)/pip-compile --generate-hashes --strip-extras --extra dev -o requirements-dev.lock pyproject.toml

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: lint
lint: ## Lint and format-check
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

.PHONY: fmt
fmt: ## Auto-format
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

.PHONY: check
check: ## Re-derive the suite's assumptions and the headline finding
	$(BIN)/erl-map check --scene $(SCENE)

.PHONY: sweep
sweep: ## Run the full detector x fusion matrix and write results/
	$(BIN)/erl-map sweep --scene $(SCENE)

.PHONY: finding
finding: ## The headline: a better detector, a worse map
	$(BIN)/erl-map run independent:0.7 --scene $(SCENE) --tag independent70
	$(BIN)/erl-map run viewbias:0.85   --scene $(SCENE) --tag viewbias85
	-$(BIN)/erl-map compare results/independent70.json results/viewbias85.json

.PHONY: docker-build
docker-build: ## Build the container image
	docker build -t $(IMAGE) .

.PHONY: docker-check
docker-check: docker-build ## Run the gate inside the container
	docker run --rm $(IMAGE)

.PHONY: docker-sweep
docker-sweep: docker-build ## Run the sweep in the container, writing to ./results
	mkdir -p results
	docker run --rm -v "$(PWD)/results:/work/results" $(IMAGE) sweep --scene conf/scenes/room_v1.yaml

.PHONY: clean
clean: ## Remove results
	rm -rf results
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean ## Also remove the virtualenv and caches
	rm -rf $(VENV) .pytest_cache .ruff_cache src/*.egg-info
