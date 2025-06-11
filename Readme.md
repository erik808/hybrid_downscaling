# TransectPicker
Let's you draw a transect in an image. The user selects point and straight paths are drawn in between. After completion the TransectPicker object will contain x,y-indices in the public `x_trans` and `y_trans` members.

## Dependencies
- `numpy`
- `matplotlib`
- `ipympl` (for local or remote use with jupyter notebooks)
- `cairo`, `pycairo` (for remote use in scripts)

See `bi_default.yml` for a working conda environment.

## Usage
button/key | action
-- | --
left mouse button  | select points
right mouse button | undo
backspace | reset
enter | exit

### Code example
A working environment is supplied in `bi_default.yml`

Examples are available in
- `test_transectpicker.py`
- `test_transectpicker.ipynb`

## Using TransectPicker (remotely) with Jupyter
This relies on `ipympl`:
```conda install -c conda-forge ipympl```

Then at the beginning of a notebook cell:
```python
%matplotlib widget
```
In cells that you don't want figures to be widgets put
```python
%matplotlib inline
```

## Using TransectPicker remotely on bi in a script
I only got this to work with cairo. So you need to install
`conda install -c conda-forge cairo pycairo`
and then set the backend before you import any other matplotlib stuff:
```python
import matplotlib
matplotlib.use('Qt5Cairo')
```
