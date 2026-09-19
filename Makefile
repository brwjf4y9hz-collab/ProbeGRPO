.PHONY: test smoke agent-smoke check-scripts check

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python3 -m probegrpo.smoke

agent-smoke:
	PYTHONPATH=src python3 -m probegrpo.agent_smoke

check-scripts:
	@for script in scripts/*.sh; do bash -n "$$script" || exit; done

check: test smoke agent-smoke check-scripts
