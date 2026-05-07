# GearLS
Source code of our DAC'26 paper: ***GearLS: Generalizable Reinforcement Learning Framework for Logic Optimization via Policy Similarity Metric*** 

![gear_frame](./GearLS_frame.png)

## Repository Structure

```
GearLS_open/
├── README.md                 
├── aigers/                   # AIGER files of benchmarks
│   ├── comb/                 # Combinational circuits converted from benchmarks for 
|   |                         # performing optimization in ABC
│   └── subcones/             # Logic cones extracted from benchmarks for RL training
├── asap7/                    # ASAP7 standard cell library files
├── output/                   # Synthesis results (empty, to be filled after experiments)
├── processaig/               # Functions for processing AIGER files faster, and script for
|                             # dataset generation and conbinational circuit conversion
├── RL/                       
│   ├── checkpoints/          # Trained model
│   ├── gearls_main.py        # Main entry file of this project, responsible for executing the 
|   |                         # GearLS optimization process.
|   └── main_train.py          # Training script for the RL agent.
├── stable_baselines3/        # Modified viersion for PSM implementation
└── syn_script/               # Yosys scripts to convert RTL benchmarks to AIGER.
```

## Environment Setup

The original platform of this project is Ubuntu 22.04 LTS.

### Dependent Packages

Please refer to the GitHub repositories for installation instructions of the following packages:

- [Berkeley-ABC.](https://github.com/berkeley-abc/abc)
- [Yosys.](https://github.com/YosysHQ/yosys)
- [AIGER utilities.](https://github.com/arminbiere/aiger)
- [OpenSTA.](https://github.com/The-OpenROAD-Project/OpenSTA)

We recommand adding the above tools to your system PATH for easier usage.

### Python Dependencies

We recommend using Conda to manage the Python environment. This project is running on **Python 3.12**.

1. The PyTorch version is 2.6.0, torch-geometric (PyG) version is 2.6.0. Install them via official instructions on their websites.

2. Please install the Python API of ABC, abc_py, manually. This work used a [forked version](https://github.com/qinys-robin/abc_py).

3. Other dependencies could be installed via pip:
```bash
pip install -r requirements.txt
```

### MobileBERT

This work uses the pre-trained encoder MobileBERT to process history commands. You can get access to this model via [HuggingFace](https://huggingface.co/google/mobilebert-uncased/). 

Download the config, pytorch model and tokenizer files, and put them in your local directory. 

## Run GearLS

### Data Generation

Benchmarks used in this work are provided in the `aigers/` dir. 

If you want to add your own benchmarks, use Yosys to convert RTLs to AIGERs and put them in `aigers/`, then modify the `processaig/transform2comb.py` and run the script to get the combinational circuits. 

You may refer to the Yosys script in `syn_script/` for the conversion process.

Make sure the design names are consistent with the AIGER file names. Before optimization, choose appropriate clock periods for your benchmarks.

### Logic Optimization using GearLS

The entrance of GearLS is `RL/gearls_main.py`. You can perform logic optimization by the following 2 ways:

#### 1. Batch Run Mode (Recommended)

Modify **line27~30** in `RL/RLEnv.py` to your local paths of ABC, Yosys, OpenSTA and MobileBERT model. These will serve as global variables in the code.

Modify **line268~272** in `RL/gearls_main.py` to specify the benchmarks you want to optimize, and the corresponding clock periods.

Then, run the following command in terminal to start the optimization process. 

```bash
cd RL
python gearls_main.py
```

#### 2. Single Circuit Run Mode (More Flexible)

You can also run GearLS by specifying arguments of your own environment and benchmarks in terminal. For example:

```bash
cd RL
python gearls_main.py --design des \
--period 600 \
--abc your/path/to/abc \
--yosys your/path/to/yosys \
--sta your/path/to/sta \
--mbert your/path/to/mobileBERT \
--device cuda:1 
```

The TNS result, optmization flow and clean execution time will also be printed in terminal. The post-mapping results will be saved in `output/`.

## Training

If you want to re-train the RL model with your own dataset:

1. Extract logic cones using `processaig/data_gen.py`. 
2. Change `RL/main_train.py` to specify the dataset path and training settings. Run **train_porcess** function first. For modifying hyper-parameters, please refer to `RL/trainingFuncs_new.py`.
3. Perform phase 1 training to get the base model:
   ```bash
   cd RL
   python main_train.py
   ```
4. Choose the best checkpoint, and change `RL/main_train.py` to run **continue_train_process** function.
5. Perform phase 2 training to get the final model.
   
## Contact and Citation

You can contact me via email: qinys2001@163.com if you have any questions (Chinese or English). 

We really appreciate it if you could cite our paper.
```
@inproceedings{qin_gearls_2026,
	title = {{GearLS}: {Generalizable} {Reinforcement} {Learning} {Framework} for {Logic} {Optimization} via {Policy} {Similarity} {Metric}},
	language = {en},
	booktitle = {2026 63rd {ACM}/{IEEE} {Design} {Automation} {Conference} ({DAC})},
	author = {Qin, Yusen and Lyu, Jiaqi and Li, Zhi and Cao, Peng},
	year = {2026},
	pages = {1--7},
}
```