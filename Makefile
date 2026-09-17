.PHONY: test smoke check

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python3 -m probegrpo.smoke

check: test smoke
