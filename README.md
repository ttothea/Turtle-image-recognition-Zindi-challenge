# Turtle Recall: Conservation Challenge


## Environment
Make sure you have the latest version of macOS (currently Monterey) installed.
Also make sure that xcode is installed and updated: 

```BASH
xcode-select --install
```

Then we can go on to install hdf5:

```BASH
 brew install hdf5
```
With the system setup like that, we can go and create our environment and install tensorflow

```BASH
pyenv local 3.9.4
python -m venv .venv
source .venv/bin/activate
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.1

pip install -U pip
pip install --no-binary=h5py h5py
pip install tensorflow-macos
pip install tensorflow-metal
pip install -r requirements.txt
```
