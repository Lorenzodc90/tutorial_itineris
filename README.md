<h1>Material for ITINERIS PhD class</h1>
L. Della Cioppa & M. Sammartino 

---

<br><br>

<p>
In this repository the necessary files for the hands-on part of this ITINERIS PhD class can be found.
This brief tutorial will instruct you to use said files.
</p>

<h2>Install miniconda</h2>

As a first step, we will install miniconda, a light-weight environment manager for python. Miniconda allows to have separate python environments with different packages installed without conflicts between them.

Donwload the appropriate version of miniconda for your computer from the [Anaconda download page](https://www.anaconda.com/download/success). Be sure to scroll down to miniconda. <br>

Although installation should be straightforward, [here](https://docs.anaconda.com/miniconda/install/) a guide can be found
<br>
If you are on windows, be sure to check the "register miniconda as my default...". 

![image](https://github.com/user-attachments/assets/f080c5cd-2fec-42d3-99b0-53e1578d5776)

If windows does not see miniconda from command line, it is necessary to add to the path folders "miniconda3" and "miniconda3/Scripts" (and sometimes "miniconda3/condabin").
Yout can adde them from windows Powershell with admin rights with the command

`[Environment]::SetEnvironmentVariable("Path", $env:Path + "; (path)", 'User')    `

where (path) is the path to add.

<h2>Setting up conda environment</h2>

Now we create the environment with the necessary packages. To check out that everything has gone right, open a command console and run the command

`conda activate base`

The conda base environment should activate.

The necessary instructions to create a working environment are contained into the YAML file "tutorial.yml". To create the environment run 

`conda env create -n [env name] -f tutorial.yml`

The creation might take a minute.

Then activate the environment with the command

`conda activate [env name]`

To deactivate a conda environment run

`conda deactivate`

Sometimes on windows system the command needs to be just

`deactivate`

(In case the creation with YAML file does not succeed, the package needed are tensorflow, netCDF4 and matplotlib. Install inside the conda env with `pip install [package]`)


<h2>Run the programs</h2>

To open a python console just run

`python`

To run a python file use the command

`python [file name]`

The files FFNN_3D_ITINERIS_TRAINING.py and LSTM_3D_ITINERIS_TRAINING.py respectively contain the Feed Forward Neural Network and LSTM codes, the file BIOARGO_MATCHUP_2018_2021.nc contains the data necessary for training.

Running the FFNN and LSTM programs will start the training of the network. Once the training is completed, the network is tested and the resulting errors are saved to images.

An in-depth documentation of the code is beyond the scope fo this readme, so look at them yourself.



