.PHONY: test smoke agent-smoke results check-scripts check

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python3 -m probegrpo.smoke

agent-smoke:
	PYTHONPATH=src python3 -m probegrpo.agent_smoke

results:
	python3 scripts/aggregate_sokoban_seeds.py \
		--csv experiments/results/public_sokoban_main_v1/per_seed.csv \
		--output-dir experiments/results/public_sokoban_main_v1
	python3 scripts/plot_sokoban_results.py \
		experiments/results/public_sokoban_main_v1/per_seed.csv \
		docs/assets/public_sokoban_main_v1.svg

check-scripts:
	@for script in scripts/*.sh; do bash -n "$$script" || exit; done

check: test smoke agent-smoke check-scripts
