## Code for MPE
This is the LaRe code for the Multi-Agent Particle Environment (MPE), based on the implementation from [STAS](https://github.com/zowiezhang/STAS). The AREL implementation is referenced from [AREL](https://github.com/baicenxiao/AREL).

### Platform and Dependencies:

- python 3.7
- pytorch 1.7.1
- gym 0.10.9
- openai
- wandb

### Install the MPE

We evaluate LaRe on six tasks from MPE, based on the implementation of [STAS](https://arxiv.org/pdf/2304.07520).
Minor modifications were made to provide individual rewards to each agent at every step, denoted by '_noshare' in the scenario names.
We also implement a competitive predator-prey task where both predators and prey are controlled by RL policies. 
Additionally, we introduce a novel task, Triangle Area, which the LLM has not encountered previously.

```shell
## Install the MPE
cd envs/multiagent-particle-envs
pip install -e .
```

### Training

```
python train_MPE.py --scenario {scenario name} --method_name {method_name}
```
**scenarios:**
| Scenario_Name              | Task Name in paper          |
|----------------------------|----------------------|
| simple_spread_n6_noshare   | CN (6 agents)        |
| simple_spread_n15_noshare  | CN (15 agents)       |
| simple_spread_n30_noshare  | CN (30 agents)       |
| simple_tag_n6_noshare      | PP (6 agents)        |
| simple_tag_n15_noshare     | PP (15 agents)       |
| simple_tag_n30_noshare     | PP (30 agents)       |
| hetero_tag_n6_noshare      | competitive PP (6 agents) |
| simple_area                | Triangle Area        |


**method_name**: ['LaRe', 'RD', 'RRD', 'STAS', 'AREL', 'PPO_dense', 'PPO'] 

