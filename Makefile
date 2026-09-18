.PHONY: test smoke check-scripts check

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python3 -m probegrpo.smoke

check-scripts:
	bash -n scripts/*.sh

check: test smoke check-scripts
