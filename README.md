# Strategically Efficient Exploration in Competitive Multi-agent Reinforcement Learning

This repository contains all code and hyperparameter configurations needed to replicate the results in [Strategically Efficient Exploration in Competitive Multi-agent Reinforcement Learning (UAI 2021)](https://arxiv.org/pdf/2107.14698.pdf)

In addition to finite Markov games, this project also supports experiments with curiosity in deep multi-agent reinforcement learning.

## Getting Started

This code has only been tested with Python 3.7.11 on Ubuntu 18.04 and 20.04 in the Windows Subsystem for Linux (WSL).

Dependencies an be installed via PIP:

```
pip install -r requirements.txt
```

This will install all the dependencies needed to reproduce published results.  Some deep RL experiment configurations us environments implemented in the [OpenSpiel](https://github.com/deepmind/open_spiel) or [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) projects, which must be installed separately.  Please refer to these projects for complete installation instructions.

## Reproducing UAI Results

Results in Figures 3 and 4 can be generated using the script "finite_games/learn_extensive_form.py" to run the appropriate training configurations:

```
cd finite_games
python learn_extensive_form.py \
    -f configs/decoy_deep_sea/strategic_ulcb.yaml \
    -f configs/decoy_deep_sea/optimistic_ulcb.yaml \
    -f configs/decoy_deep_sea/strategic_nash_q.yaml \
    -f configs/decoy_deep_sea/optimistic_nash_q.yaml
```

Experiment configurations can be run separately if preferred.  Results for Figure 5 can be generated using:

```
python learn_extensive_form.py \
    -f configs/alpha_beta/strategic_ulcb.yaml \
    -f configs/alpha_beta/optimistic_ulcb.yaml \
    -f configs/alpha_beta/strategic_nash_q.yaml \
    -f configs/alpha_beta/optimistic_nash_q.yaml
```

Figures can be generated using the "finite_games/plot_runs.py" script.  Note that this script requires

Example:

```
python plot_runs.py \
    "Strategic ULCB" results/debug/decoy_deep_sea_strategic_ulcb/decoy_deep_sea_strategic_ulcb_decoy_games=50,decoy_size=20 \
    "Optimistic ULCB" results/debug/decoy_deep_sea_optimistic_ulcb/decoy_deep_sea_optimistic_ulcb_decoy_games=50,decoy_size=20,exploit=True
```

## Deep RL Experiments

Deep RL experiments use [RLLib](https://github.com/ray-project/ray/tree/master/rllib) 0.8.3 and [Tensorflow](https://www.tensorflow.org/) 2.4.2, both installed by "requirements.txt".  Experiments with deep multi-agent RL can be run with the "train_multiagent.py" script.

Example:

```
python3 train_multiagent.py -f experiment_configs/roshambo/ppo_hybrid_bandit.yaml --nash-conv
```

This will train [PPO](https://arxiv.org/abs/1707.06347) in self-play in a simple two-player matrix game.  This project currently supports two intrinsic reward mechansims with multi-agent PPO, [Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf) and the [Intrinsic Curiosity Module](https://arxiv.org/abs/1705.05363). 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
