.PHONY: help install install-train data train eval export enrich drift test clean

PY ?= python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install core + dev (CPU; covers data synthesis, metrics, enrichment, tests)
	$(PY) -m pip install -e ".[dev]"

install-train: ## Install the from-scratch training + ONNX stack (torch, onnx, onnxruntime)
	$(PY) -m pip install -e ".[train,dev]"

data: ## Generate the train/test frame splits deterministically
	$(PY) -m constellation_vision.data.generate

train: ## Train the UNet-lite from scratch, export ONNX, evaluate, write artifacts/EVAL.md
	$(PY) -m constellation_vision.train

eval: ## Evaluate the committed checkpoint on the held-out split -> artifacts/EVAL.md
	$(PY) -m constellation_vision.eval.harness

export: ## Export the trained checkpoint to ONNX
	$(PY) -m constellation_vision.model.export_onnx

enrich: ## Run the argus enrichment step on a synthetic demo frame
	$(PY) -m constellation_vision.enrich.argus_enrich

drift: ## Drift-check the synthetic frame distribution against committed stats
	$(PY) -m constellation_vision.data.drift

test: ## Tests + quality gate (metrics calibration + coverage floor)
	$(PY) -m pytest --cov=constellation_vision --cov-report=term-missing --cov-fail-under=85

clean: ## Remove generated data, artifacts, and weights
	rm -rf data/*.npz artifacts/metrics.json .pytest_cache .coverage \
		src/constellation_vision/model/weights src/*.egg-info
