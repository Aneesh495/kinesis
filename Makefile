PYTHON ?= .venv/bin/python
PIP_INSTALL ?= uv pip install --python $(PYTHON)
RUN_ID ?= default_run
SEED ?= 42
TASK ?= cartpole_swingup
EVIDENCE ?= evidence/$(RUN_ID)

.PHONY: help bootstrap doctor build test test-control collect test-replay plan closed-loop shift-study data train evaluate ablate stress bench docs demo acceptance verify release clean

help:
	@echo "Kinesis: Learned world models for partially observed control"
	@echo "Targets:"
	@echo "  bootstrap    - Set up dependencies and virtual environment"
	@echo "  doctor       - Check environment, hardware, precision, and toolchain"
	@echo "  build        - Install package in editable mode"
	@echo "  test         - Run unit and semantic test suites"
	@echo "  test-control - Run analytic linear control benchmark tests"
	@echo "  test-replay  - Run replay buffer integrity and recovery tests"
	@echo "  collect      - Run environment collection routine"
	@echo "  plan         - Run latent planning test verification"
	@echo "  closed-loop  - Run closed-loop execution tests"
	@echo "  shift-study  - Run parameter shift evaluation suite"
	@echo "  data         - Acquire/validate dataset manifests"
	@echo "  train        - Execute training run (TASK=$(TASK) SEED=$(SEED))"
	@echo "  evaluate     - Evaluate frozen run across test environments"
	@echo "  ablate       - Execute ablation study matrix"
	@echo "  stress       - Run stress, fault, and interruption recovery tests"
	@echo "  bench        - Run throughput and latency benchmarks"
	@echo "  docs         - Check documentation, diagrams, and formatting"
	@echo "  demo         - Run closed-loop demonstration"
	@echo "  acceptance   - Generate complete acceptance evidence bundle (RUN_ID=$(RUN_ID))"
	@echo "  verify       - Independently verify acceptance evidence (EVIDENCE=$(EVIDENCE))"
	@echo "  release      - Package release artifacts"
	@echo "  clean        - Remove temporary files and caches"

bootstrap:
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python -e ".[dev]"

doctor:
	$(PYTHON) -m kinesis.doctor

build:
	$(PIP_INSTALL) -e .

test:
	$(PYTHON) -m pytest tests/ -v

test-control:
	$(PYTHON) -m pytest tests/test_analytic_control.py -v

test-replay:
	$(PYTHON) -m pytest tests/test_replay.py -v

collect:
	$(PYTHON) -m kinesis.cli collect --task $(TASK) --episodes 10 --seed $(SEED)

plan:
	$(PYTHON) -m pytest tests/test_planning.py -v

closed-loop:
	$(PYTHON) -m pytest tests/test_closed_loop.py -v

shift-study:
	$(PYTHON) -m kinesis.cli evaluate --task $(TASK) --shift-study --seed $(SEED)

data:
	$(PYTHON) -m kinesis.cli data --validate

train:
	$(PYTHON) -m kinesis.cli train --task $(TASK) --seed $(SEED) --run-id $(RUN_ID)

evaluate:
	$(PYTHON) -m kinesis.cli evaluate --run-id $(RUN_ID)

ablate:
	$(PYTHON) -m kinesis.cli ablate --task $(TASK) --seed $(SEED)

stress:
	$(PYTHON) -m pytest tests/test_stress.py -v

bench:
	$(PYTHON) -m kinesis.cli bench

docs:
	$(PYTHON) -m kinesis.cli check-docs

demo:
	$(PYTHON) -m kinesis.cli demo --task $(TASK) --seed $(SEED)

acceptance:
	$(PYTHON) -m kinesis.artifacts.acceptance --run-id $(RUN_ID)

verify:
	$(PYTHON) -m kinesis.artifacts.verifier --evidence $(EVIDENCE)

release:
	$(PYTHON) -m kinesis.cli release

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
