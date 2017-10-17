# Setup



## Activate proper ROOT environment

```bash
source /afs/cern.ch/sw/lcg/contrib/gcc/4.3/x86_64-slc5/setup.sh &&cd /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.00/x86_64-slc5-gcc43-opt/root &&source bin/thisroot.sh &&cd -
```


## Setup `virtuaenv`


```bash
RPYTHON = /afs/cern.ch/sw/lcg/external/Python/2.7.3/x86_64-slc6-gcc46-opt/bin/python
virtualenv -p $RPYTHON .env
source .env/bin/activate


# Check if everything is ok
which python
python -c "import ROOT"
pip --version

# Upgrade pip
pip install pip -- upgrade
```

## Install requirements

```bash
# Create requirements
# NB: keep requirements.txt inside your project folder
pip install -r requirements.txt
```