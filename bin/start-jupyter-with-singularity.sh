#!/bin/bash

# change direcory to user scratch directory
cd /scratch/$USER

# Check if the directory exists
if [ -d "./Generative-AI-with-LLMs" ]; then
  rm -rf "./Generative-AI-with-LLMs"
fi

# clone the GenAI GitHub repository
git clone https://github.com/hwang2006/Generative-AI-with-LLMs.git

# change directory to the GenAI directory that you have just cloned.
cd Generative-AI-with-LLMs

# the working directory of Jupyter Notebook to the GenAI directory 
sed -i 's/cd \/scratch\/\$USER/cd \/scratch\/\$USER\/Generative-AI-with-LLMs/g' ./bin/jupyter_run_singularity.sh 

# submit the job script and launch the jupyter server on a compute node  
sbatch ./bin/jupyter_run_singularity.sh

sleep 3

echo "Jupyter Server is running at:" 
cat port_forwarding_command
