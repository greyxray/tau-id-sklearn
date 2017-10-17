virtualenv -p /afs/cern.ch/sw/lcg/external/Python/2.7.3/x86_64-slc6-gcc46-opt/bin/python .env
source .env/bin/activate


# Check if everything is ok
which python
python -c "import ROOT"
pip --version

# Upgrade pip
pip install pip -- upgrade

# Create requirements
# NB: keep requirements.txt inside your project folder

echo numpy >> requirements.txt
echo scipy >> requirements.txt
echo sklearn >> requirements.txt
echo rootpy >> requirements.txt

# This one fails
# echo root_numpy >> requirements.txt

# Install dependencies
pip install -r requirements.txt
