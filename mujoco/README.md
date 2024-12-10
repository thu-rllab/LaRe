## Code for MuJoCo
The LaRe code for the MuJoCo locomotion benchmark is based on the implementation of TD3. 
We utilize tasks provided by [Gymnasium](https://gymnasium.farama.org/environments/mujoco/). 
The implementations of SAC and PPO are adapted from [DRL-code-pytorch](https://github.com/Lizhi-sjtu/DRL-code-pytorch).

### Installation

```shell
# python 3.8.19
pip install -r requirements.txt
```

### Training

```
python main.py --policy={policy} --env={env} --rd_method={rd_method}
```

**policy**: [TD3, SAC, DDPG]. If using PPO as the policy, execute `ppo_main.py` instead.

**env**: [Reacher-v4, HalfCheetah-v4, Walker2d-v4, HumanoidStandup-v4]

**rd_method**: [LaRe_RRDu, LaRe_RD, RD, RRD, RRD_unbiased, IRCR, VIB, Diaster]




