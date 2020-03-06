This is the code for the paper "On Concept-Based Explanations in Deep Neural
Networks" https://arxiv.org/abs/1910.07969".

1.  Please download all data from "gs://concept_discovery/data/" or
    "https://console.cloud.google.com/storage/browser/concept_discovery", and
    change the directory paths when loading data.
2.  To run the toy example, just run python3 toy_main.py.
3.  To create the toy example, just run python3 create_toy.py.
4.  To run the AwA example, just run python3 awa_main.py.

ipca.py is a general helper function for calculating the completeness of a given
model, and toy_helper.py and awa_helper.py contain helper functions that are
specific to the datasets.




////// Tony's note

Installation:

# first time installation
cd your_directory/intuit-project
virtualenv -p python3.6 .
source ./bin/activate
cd concept_explanations
pip install -r requirements.txt # if do not have gpu, see line 29
pip install IPython # just for debugging

# Every time you try to run the code
cd your_directory/intuit-project
source ./bin/activate
export PYTHONPATH=$PYTHONPATH:/your_directory/intuit-project # notice to use full path here

# To run toy example
1. Run python3 create_toy.py, which creates the dataset
   - Only need to run once, it will save the dataset as .npy files
2. Run python3 toy_main.py

if using GPU might want to add CUDA_VISIBLE_DEVICES=0 before python3


# trouble shooting
error:
module 'tensorflow' has no attribute 'name_scope'
do:
pip install --upgrade pip setuptools wheel
pip install -I tensorflow-gpu==1.14
pip install -I keras

error:
Loaded runtime CuDNN library: 7.0.5 but source was compiled with: 7.2.1.
do:
pip3 uninstall tensorflow-gpu
pip3 install tensorflow-gpu==1.9.0

If things still does not work, delete all files virtualenv creates and start from scratch again


# please ignore, for tony's system specifically

cd /home/ericwallace/tonyzhaozh/Projects/intuit-project
source ./bin/activate
export PYTHONPATH=$PYTHONPATH:/home/ericwallace/tonyzhaozh/Projects/intuit-project
cd concept_explanations/

CUDA_VISIBLE_DEVICES=0 python3 create_toy.py

<!--export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64/-->
<!--export PATH=$PATH:/usr/local/cuda/bin/-->

<!--export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/lib:/usr/lib64-->


