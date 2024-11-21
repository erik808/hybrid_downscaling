## Tests:
After cloning repo, run the tests:
`pytest -sv test_everything.py`

## Main experiment:
- Check configuration: `configs/default.py`
- Tweak contents of `if==__main__` block in `ae_experiment.py`
- Run `python ae_experiment.py`

## Parallel use of optuna:
`cd parallel`
`./wrap_submit.sh <exec>`

