# In-depth Understanding of CLIP Neurons via Faceted Feature Visualization
Hyesu Lim, Changdae Oh, Junhyeok Park, Rohan Prasad

> In this project, we reproduce the result of multimodal neuron analysis (OpenAI, 2021) and further investigate the behavior of such neurons after being fine-tuned on diverse downstream task.


<br>

## Initial Setup
1. Prepare the basic experiment pipeline by cloning this repository.
``` shell
git clone https://github.com/hyesulim/clip-feat-vis.git
cd clip-feat-vis
```
2. Environment setup (any of python ver >= 3.7, torch ver >= 1.8 should be okay). However, this code has been extensively tested on Python 3.11.6 and PyTorch 2.1.1 and is what we recommend. A `requirements.txt` file is included to help resolve any version conflicts.
``` shell
conda create -n mmn python=3.11
conda activate mmn
pip install -r requirements.txt
```
3. A sample Jupyter notebook is included inside the `notebooks`. This should help you get started.

<br>

## Module description

1. `linear_probe` - This folder contains all our code related to training the Linear Probe used for faceted visualization
2. `initial_ablations' - This folder contains the scripts and codes that used Lucent for performing the initial ablations on vanilla feature visualization
3. `faceted_visualization` - This folder contains all the experiments and code related to faceted visualization

   a.) `visualizer` - this is a python package containing reusable code for using faceted visualization in your project. It currently only supports CLIP models.
4. finetuning - This folder contains our the code required for finetuning

### Acknowledgement
_This repository is built on top of [Lucent](https://github.com/greentfrapp/lucent) library, and we would like to 
thank the authors of Lucent for their extensive efforts.!_
