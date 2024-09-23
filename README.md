# Generarive AI with LLMs Practices on a Supercomputer

Generative AI with LLMs refers to the use of large language models like GPT-3 for generating human-like content, spanning text, images and even code. LLMs are trained on a vast amount of data and code, and usually carefully prompt-engineered or fine-tuned to suit specific downstream tasks such as Chatbots, Translation, Question Answering and Summarization. 

This repository is intended to share and promote best practices for Generative AI using LLMs on a supercomputer. The python codes in the Lab exercises are sourced from the 16-hour [Generative AI with LLMs Online Course](https://www.coursera.org/learn/generative-ai-with-llms) offered by the DeepLearning.AI. 

**Contents**
* [KISTI Neuron GPU Cluster](#kisti-neuron-gpu-cluster)
* [Installing Conda](#installing-conda)
* [Creating a Conda Virtual Environment](#creating-a-conda-virtual-environment)
* [Running Jupyter](#running-jupyter)
* [Building a Singularity Container Image](#building-a-singularity-container-image)
* [Running Jupyter Using a Singularity Container Image for Generative AI Practices](#running-jupyter-with-a-singularity-container-image-for-generative-ai-practices)
* [Lab Exercises with a QuickStart Guide](#lab-exercises-with-a-quickstart-guide)
* [Reference](#reference)


## KISTI Neuron GPU Cluster
Neuron is a KISTI GPU cluster system consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

1. Check the Neuron system specification
```
[glogin01]$ cat /etc/*release*
CentOS Linux release 7.9.2009 (Core)
Derived from Red Hat Enterprise Linux 7.8 (Source)
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"

CentOS Linux release 7.9.2009 (Core)
CentOS Linux release 7.9.2009 (Core)
cpe:/o:centos:centos:7
```

2. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate
```
```
# (option 2) Miniconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
```

3. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. You will install and create subsequent conda environments on your scratch directory. 
```
[glogin01]$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
[glogin01]$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3  <======== type /scratch/$USER/miniconda3 here
PREFIX=/scratch/qualis/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish to update your shell profile to automatically initialize conda?
This will activate conda on startup and change the command prompt when activated.
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
no change     /scratch/qualis/miniconda3/etc/profile.d/conda.csh
modified      /home01/qualis/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

Thank you for installing Miniconda3!
```

4. finalize installing Miniconda with environment variables set including conda path

```
[glogin01]$ source ~/.bashrc    # set conda path and environment variables 
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 23.9.0
```

## Creating a Conda Virtual Environment
You want to create a virtual envrionment with a python version 3.10 for Generative AI Practices.
```
[glogin01]$ conda create -n genai python=3.10
Retrieving notices: ...working... done
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3/envs/genai

  added / updated specs:
    - python=3.10
.
.
.
Proceed ([y]/n)? y    <========== type yes 


Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate genai
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

## Running Jupyter
[Jupyter](https://jupyter.org/) is free software, open standards, and web services for interactive computing across all programming languages. Jupyterlab is the latest web-based interactive development environment for notebooks, code, and data. The Jupyter Notebook is the original web application for creating and sharing computational documents. You will run a notebook server on a worker node (*not* on a login node), which will be accessed from the browser on your PC or labtop through SSH tunneling. 
<p align="center"><img src="https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/assets/84169368/34a753fc-ccb7-423e-b0f3-f973b8cd7122"/>
</p>

In order to do so, you need to add the "genai" virtual envrionment that you have created as a python kernel.
1. activate the `genai` virtual environment:
```
[glogin01]$ conda activate genai
```
2. install Jupyter on the virtual environment:
```
(genai) [glogin01]$ conda install jupyter chardet cchardet 
  
```
3. add the virtual environment as a jupyter kernel:
```
(genai) [glogin01]$ pip install ipykernel; python -m ipykernel install --user --name genai
```
4. check the list of kernels currently installed:
```
(genai) [glogin01]$ jupyter kernelspec list
Available kernels:
python3       /home01/$USER/.local/share/jupyter/kernels/python3
genai         /home01/$USER/.local/share/jupyter/kernels/genai
```
5. launch a jupyter notebook server on a worker node 
- to deactivate the virtual environment
```
(genai) [glogin01]$ conda deactivate
```
- to create a batch script for launching a jupyter notebook server: 
```
[glogin01]$ cat jupyter_run.sh
#!/bin/bash
#SBATCH --comment=pytorch
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=cas_v100nv_8
##SBATCH --partition=cas_v100_4
#SBATCH --time=12:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=8     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the port and node name
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
module load gcc/10.2.0 cuda/12.3

echo "execute jupyter"
source ~/.bashrc
conda activate genai
cd /scratch/$USER  # the root/work directory of Jupyter lab/notebook
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} #jupyter token: your account ID
echo "end of the job"
```
- to launch a jupyter notebook server 
```
[glogin01]$ sbatch jupyter_run.sh
Submitted batch job XXXXXX
```
- to check if the jupyter notebook server is up and running
```
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    amd_a100nv_8 jupyter_    $USER  RUNNING       0:02   8:00:00      1 gpu30
[glogin01]$ cat slurm-XXXXXX.out
.
.
[I 2023-02-14 08:30:04.790 ServerApp] Jupyter Server 1.23.4 is running at:
[I 2023-02-14 08:30:04.790 ServerApp] http://gpu##:#####/lab?token=...
.
.
```
- to check the SSH tunneling information generated by the jupyter_run.sh script 
```
[glogin01]$ cat port_forwarding_command
ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
```
6. open a new SSH client (e.g., Putty, MobaXterm, PowerShell, Command Prompt, etc) on your PC or laptop and log in to the Neuron system just by copying and pasting the port_forwarding_command:

![20240123_102609](https://github.com/hwang2006/Generative-AI-with-LLMs/assets/84169368/1f5dd57f-9872-491b-8dd4-0aa99b867789)

7. open a web browser on your PC or laptop to access the jupyter server
```
URL Address: localhost:8888
Password or token: $USER    # your account name on Neuron
```
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p> 

## Building a Singularity Container Image
You can build your a singularity container image for Generativ AI. In order to build a singularity container on Neuron, you need to have a fakeroot permission that you can obtain by requesting it to the system administrator.  
```
# create a Singularity recipe file
[glogin01]$ cat genai.def
bootstrap: docker
from: nvcr.io/nvidia/pytorch:22.09-py3
%post
echo "Conda installing Jupyter..."
conda update --all
conda install python=3.10
conda update --all
conda install jupyter chardet cchardet -y
conda install -c conda-forge jupytext -y
echo "PIP installing torchdata transformers datasets"
pip install torch==1.13.0 torchdata transformers datasets
echo "PIP installing evaluate rouge_score loralib peft"
pip install evaluate rouge_score loralib peft
echo "PIP tri..."
pip install git+https://github.com/lvwerra/trl.git@25fa1bd

# build a container image
[glogin01]$ singularity build --fakeroot GenAI.sif genai.def
```
## Running Jupyter with a Singularity Container Image for Generative AI Practices
You can launch a Jupyter server using the GenAI container image that you have created by submitting and running it on a compute node. You can then access it through the SSH tunneling mechanizm by opening a browser on your PC or labtop. Please be aware that with the Singularity container image, there is no need to install the Miniconda3 on your scratch directory and build the conda virtual environment for Generative AI practices. 
- create a batch script for launching a jupyter notebook server. We assume that you have the Singularity container image called "GenAI.sif" available at your hands. Or, you can have access to the "genai-pytorch:22.09-py3.sif" cotainer image that is available in the "/apps/applications/singularity_images/ngc" directory on the Neuron system. Note that the "sed    
```
[glogin01]$  cat jupyter_run_singularity.sh
#!/bin/bash
#SBATCH --comment=pytorch
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=cas_v100nv_8
##SBATCH --partition=cas_v100_4
#SBATCH --time=12:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=4     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the port and node name
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
module load singularity/3.9.7

echo "execute jupyter"
cd /scratch/$USER  # the root/work directory of Jupyter lab/notebook
singularity run --nv /apps/applications/singularity_images/ngc/genai-pytorch:22.09-py3.sif jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} #jupyter token: your account ID
#singularity run --nv GenAI.sif jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} #jupyter token: your account ID
echo "end of the job"
```
- launch a jupyter server by submitting the batch script to a worker node. 
```
[glogin01]$ sbatch jupyter_run_singularity.sh
Submitted batch job XXXXXX
```
- check if the jupyter server is up and running
```
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    amd_a100nv_8 jupyter_    $USER  RUNNING       0:02   8:00:00      1 gpu30
```

## Lab Exercises with a QuickStart Guide
Now, you are ready to do Generative AI with LLMs practices either using the *genai* conda environment that you have created or the *genai* container image. You may want to clone this GitHub repository on your scratch directory (e.g., /scratch/$USER), and you should able to see the lab exercises jupyter notebook codes via the Jupyter Notebook interface that you have launched. You could start with *Lab_1_summarize_dialogue.ipynb* just by clickihng it that covers prompting and prompt engineering practices. Instruction and LoRA PEFT fine-tunings are discussed in Lab_2 and RLHF practices are in Lab_3. 

Here is a **QuickStart guide** that you can copy and paste that lets you jump right in hands-on lab exercises for Generative AI with LLMs on Neuron, no conda installation and the conda virtual environment required. Simply leverage the pre-built Singularity *genai* container image, readily available in the /apps/applications/singularity_images/ngc directory.

Let's assume that you have been logged on in the Neuron system.
```
[glogin01]$ cd /scratch/$USER
[glogin01]$ git clone https://github.com/hwang2006/Generative-AI-with-LLMs.git
[glogin01]$ cd Generative-AI-with-LLMs
[glogin01]$ ls
./     doc/                                Lab_2_fine_tune_generative_ai_model.ipynb
../    flan-t5-samsum-summarization.ipynb  Lab_3_fine_tune_model_to_detoxify_summaries.ipynb
bin/   .git/                               README.md
data/  Lab_1_summarize_dialogue.ipynb      singularity/

[glogin01]$ sed -i 's/cd \/scratch\/\$USER/cd \/scratch\/\$USER\/Generative-AI-with-LLMs/g' ./bin/jupyter_run_singularity.sh 

[glogin01]$ cat ./bin/jupyter_run_singularity.sh
#!/bin/bash
#SBATCH --comment=pytorch
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=cas_v100nv_8
##SBATCH --partition=cas_v100_4
##SBATCH --partition=edu
#SBATCH --time=12:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=4     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the port and node name
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
module load singularity/3.9.7

echo "execute jupyter"
cd /scratch/$USER/Generative-AI-with-LLMs  # the root/work directory of Jupyter lab/notebook
singularity run --nv /apps/applications/singularity_images/ngc/genai-pytorch:22.09-py3.sif jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} #jupyter token: your account ID
#singularity run --nv GenAI.sif jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} #jupyter token: your account ID
echo "end of the job"

[glogin01]$ sbatch ./bin/jupyter_run_singularity.sh
Submitted batch job XXXXXX

[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    amd_a100nv_8 jupyter_    $USER  RUNNING       0:02   8:00:00      1 gpu##

[glogin01]$ cat port_forwarding_command
ssh -L localhost:8888:gpu##:##### your-account-name@neuron.ksc.re.kr
```
Note that the "sed -i 's/cd \/scratch\/\$USER/...." command above is to replace "cd /scratch/$USER" with "cd /scratch/$USER/Generative-AI-with-LLMs" in the jupyter_run_singularity.sh script, aiming to change the working directory of Jupyter Notebook to the git directory that you have cloned. You may also notice that the partition is set to be "amd_a100nv_8" in the script that you may want to switch to different partitions (e.g., cas_v100_4) depending on idle nodes availability. Please refer to the shell script, "start-jupyter-with-singularity.sh" in the bin directory of this GitHub repository.   

Once the jupyter server is up and running on a computer node, you can open a new terminal to make a SSH client connection using the port_forwarding_command and then open a web browser to launch a Jupyter client interface as described in the last part of the [Running Jupyter](#running-jupyter) section.  

![20240123_092630](https://github.com/hwang2006/Generative-AI-with-LLMs/assets/84169368/d8a291bb-65e0-4c2f-a9d8-bc724fbb7034)


## Reference
[[DeepLearning.AI Online Course] Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms)  
[Generative AI with LLMs Practices on Perlmutter at LBNL/NERSC](https://github.com/hwang2006/Generative-AI-with-LLMs-Practices-on-Perlmutter)  


