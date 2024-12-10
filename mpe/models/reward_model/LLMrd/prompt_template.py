# from envs import REGISTRY as env_REGISTRY
import json
import numpy as np
import sys
sys.path.append('/home/quy/STAS')

factor_role_instrutor = f"ROLE INSTRUCTION: You are good at understanding tasks and writing python codes.\
You should fully understand the provided task and describe the exact observation form in the current map. \
Then, based on your understanding, analyze potential positive and negative behaviours or statuses that can be reflected in the observation.\
Finally, write an evaluation function that returns factors evaluating the current status from different aspects. \
Note:1. Do not use information you are not given! \
2. Focus on the most relevant evaluation factors and use information in observation as little as possible. \
3. The code should be as generic, complete and not contain omissions! \
4. Avoid dividing by zero!\
5. The input variable is in the form of (batch_size, dim), please return a list of several evaluation factor arrays, each in the form of (batch_size, 1). \
Please think step by step and adhere to the following JSON format (just replace the () with your answer):"+\
"{\
Understand: (give your thought about the exact observation form in current map),\
Analyze: (think step by step and analyze potential positive and negative behaviors or statuses that can be reflected in observations), \
Functions: (a python function with the form of 'def evaluation_func(observation): ... return [a list of evaluation factor arrays]')\
}"

class Base_prompt(object):
    def __init__(self, map_name, factor_decomp=False) -> None:
        self.map_name = map_name
        self.factor_decomp = factor_decomp
        self.task_description = ''
        self.state_form = ''
        self.role_instruction = ''

    def get_message(self):

        message=[]
        message.append({'role':'user','content':self.task_description+self.state_form+self.role_instruction})
        return message

    def factor_check(self, out_content):
        error_idx, error_content = -1, ''
        pass_check = True
        factor_num = 0
        for i in range(len(out_content)):
            try:
                func = json.loads(out_content[i])['Functions']
                namespace = {}
                exec(func,namespace)
                active_evaluation_func = namespace['evaluation_func']
                evaluation_factors = active_evaluation_func(np.stack(self.obs, axis=0))
                factor_num = len(evaluation_factors)
                for factor in evaluation_factors:
                    if len(factor.shape) != 2 or factor.shape[0] != self.n_agents or factor.shape[1] != 1:
                        pass_check=False
                        error_idx = i
                        error_content = f'There is an error in your previous answer. Error: The shape of the output factors should be (batch_size, 1).'
            except Exception as e:
                pass_check=False
                error_idx = i
                error_content = f'There is an error in your previous answer. Error:{e.args}' 
        return pass_check, error_idx, error_content, factor_num


class MPE_simple_spread_prompt(Base_prompt):
    def __init__(self, map_name, factor_decomp=False) -> None:
        super().__init__(map_name, factor_decomp)
        from utils.util import setup_seed, make_env, n_actions
        self.env = make_env(map_name)
        self.n_others = self.env.world.n_others
        self.obs = self.env.reset()
        self.n_agents = len(self.env.world.agents)
        self.n_landmarks = len(self.env.world.landmarks)
        self.agent_size = self.env.world.agent_size
        self.collision_size = 2 * self.agent_size

        self.task_description = f"TASK:We are playing Cooperative Navigation. In this task, {self.n_agents} agents are asked to reach {self.n_landmarks} landmarks. \
Each agent should move to get close to one landmark and avoid collision(distance<={self.collision_size}) with other agents.\n"
        self.state_form = f"OBSERVATION FORM: At each step, each agent receives an observation array.\
This observation is represented by an array with {len(self.obs[0])} dimensions:\
concat([agent's velocity, agent's location, {self.n_others+1} nearest landmarks' relative locations, {self.n_others} nearest other agents' relative locations]).\
The agent's velocity is 2-dimensional, including x and y components.\
The agent's location is 2-dimensional, including x and y coordinates.\
The landmarks' relative locations are {self.n_others+1}x2 dimensions, including relative x and y coordinates(landmark_i.x-agent.x, landmark_i.y-agent.y).\
The other agents' relative locations are {self.n_others}x2 dimensions, including relative x and y coordinates(agent_i.x-agent.x, agent_i.y-agent.y).\n"

        self.role_instruction = factor_role_instrutor
  
class MPE_simple_tag_prompt(Base_prompt):
    def __init__(self, map_name, factor_decomp=False) -> None:
        super().__init__(map_name, factor_decomp)
        from utils.util import setup_seed, make_env, n_actions
        self.env = make_env(map_name)
        self.obs = self.env.reset()
        self.n_agents = self.env.world.num_adversaries
        self.n_preys = len(self.env.world.agents) - self.n_agents
        self.n_adv_visible_agent = self.env.world.n_adv_visible_agent
        self.n_adv_visible_landmark = self.env.world.n_adv_visible_landmark
        self.n_adv_visible_adv = self.env.world.n_adv_visible_adv
        self.agent_size = self.env.world.agent_size
        self.adversary_size = self.env.world.adversary_size
        self.collision_dis = self.agent_size + self.adversary_size

        self.task_description = f"TASK:We are playing Predator Prey. In this task, {self.n_agents} agents should cooperate to catch {self.n_preys} preys. \
Agents should move to get closer to one of the preys to catch it(distance<{self.collision_dis}). Landmarks represent only fixed obstacles.\n"
        self.state_form = f"OBSERVATION FORM: At each step, each agent receives an observation array.\
This observation is represented by an array with {len(self.obs[0])} dimensions:\
concat([agent's velocity, agent's location, {self.n_adv_visible_landmark} nearest landmarks' relative locations, \
{self.n_adv_visible_adv} nearest other agents' relative locations, {self.n_adv_visible_agent} nearest preys' relative locations, \
{self.n_adv_visible_agent} nearest preys' velocity]).\
The agent's velocity is 2-dimensional, including x and y components.\
The agent's location is 2-dimensional, including x and y coordinates.\
The landmarks' relative locations are {self.n_adv_visible_landmark}x2 dimensions, including relative x and y coordinates(landmark_i.x-agent.x, landmark_i.y-agent.y).\
The other agents' relative locations are {self.n_adv_visible_adv}x2 dimensions, including relative x and y coordinates(agent_i.x-agent.x, agent_i.y-agent.y).\
The preys' relative locations are {self.n_adv_visible_agent}x2-dimensional, including relative x and y coordinates(prey_i.x-agent.x, prey_i.y-agent.y).\
The preys' velocity is {self.n_adv_visible_agent}x2-dimensional, including x and y components.\n"
        self.role_instruction = factor_role_instrutor

    
class MPE_hetero_tag_prompt(Base_prompt):
    def __init__(self, map_name='hetero_tag_n6_noshare', factor_decomp=False) -> None:
        super().__init__(map_name, factor_decomp)
        from utils.util import setup_seed, make_env, n_actions
        self.env = make_env(map_name)
        self.obs = self.env.reset()
        self.n_advs = self.env.world.num_adversaries
        self.n_preys = len(self.env.world.agents) - self.n_advs
        self.adv_obs = self.obs[:self.n_advs]
        self.prey_obs = self.obs[self.n_advs:]
        self.n_adv_visible_agent = self.env.world.n_adv_visible_agent
        self.n_adv_visible_landmark = self.env.world.n_adv_visible_landmark
        self.n_adv_visible_adv = self.env.world.n_adv_visible_adv
        self.n_visible_agent = self.env.world.n_visible_agent
        self.n_visible_landmark = self.env.world.n_visible_landmark
        self.n_visible_adv = self.env.world.n_visible_adv
        self.agent_size = self.env.world.agent_size
        self.adversary_size = self.env.world.adversary_size
        self.collision_dis = self.agent_size + self.adversary_size

        self.adv_task_description = f"TASK:We are playing a Predator in the game Predator-Prey. \
In this task, {self.n_advs} agents should cooperate to catch {self.n_preys} preys. \
Agents should move to get closer to one of the preys to catch it(distance<{self.collision_dis}). \
Landmarks represent only fixed obstacles.\n"
        self.adv_state_form = f"OBSERVATION FORM: At each step, each agent receives an observation array.\
This observation is represented by an array with {len(self.adv_obs[0])} dimensions:\
concat([agent's velocity, agent's location, {self.n_adv_visible_landmark} nearest landmarks' relative locations, \
{self.n_adv_visible_adv} nearest other agents' relative locations, {self.n_adv_visible_agent} nearest preys' relative locations, \
{self.n_adv_visible_agent} nearest preys' velocity]).\
The agent's velocity is 2-dimensional, including x and y components.\
The agent's location is 2-dimensional, including x and y coordinates.\
The landmarks' relative locations are {self.n_adv_visible_landmark}x2 dimensions, including relative x and y coordinates(landmark_i.x-agent.x, landmark_i.y-agent.y).\
The other agents' relative locations are {self.n_adv_visible_adv}x2 dimensions, including relative x and y coordinates(agent_i.x-agent.x, agent_i.y-agent.y).\
The preys' relative locations are {self.n_adv_visible_agent}x2-dimensional, including relative x and y coordinates(prey_i.x-agent.x, prey_i.y-agent.y).\
The preys' velocity is {self.n_adv_visible_agent}x2-dimensional, including x and y components.\n"
        
        self.prey_task_description = f"TASK:We are playing a Prey in the game Predator-Prey. \
In this task, {self.n_preys} agents should try to escape from  all of the {self.n_advs} predators. \
Agents should move to get away from all of the predators to avoid being caught(distance<{self.collision_dis}). \
Landmarks represent only fixed obstacles.\n"
        self.prey_state_form = f"OBSERVATION FORM: At each step, each agent receives an observation array.\
This observation is represented by an array with {len(self.prey_obs[0])} dimensions:\
concat([agent's velocity, agent's location, {self.n_visible_landmark} nearest landmarks' relative locations, \
{self.n_visible_adv} nearest predators' relative locations, {self.n_visible_agent} nearest other preys' relative locations, \
{self.n_visible_agent} nearest preys' velocity]).\
The agent's velocity is 2-dimensional, including x and y components.\
The agent's location is 2-dimensional, including x and y coordinates.\
The landmarks' relative locations are {self.n_visible_landmark}x2 dimensions, including relative x and y coordinates(landmark_i.x-agent.x, landmark_i.y-agent.y).\
The predators' relative locations are {self.n_visible_adv}x2 dimensions, including relative x and y coordinates(predator_i.x-agent.x, predator_i.y-agent.y).\
The other preys' relative locations are {self.n_visible_agent}x2-dimensional, including relative x and y coordinates(prey_i.x-agent.x, prey_i.y-agent.y).\
The other preys' velocity is {self.n_visible_agent}x2-dimensional, including x and y components.\n"
        
        self.role_instruction = factor_role_instrutor
    def get_message(self):
        adv_message=[]
        adv_message.append({'role':'user','content':self.adv_task_description+self.adv_state_form+self.role_instruction})
        prey_message=[]
        prey_message.append({'role':'user','content':self.prey_task_description+self.prey_state_form+self.role_instruction})
        return [adv_message, prey_message]
    
    def factor_check(self, out_content, id=0):
        error_idx, error_content = -1, ''
        pass_check = True
        factor_num = 0
        local_obs = self.adv_obs if id==0 else self.prey_obs
        for i in range(len(out_content)):
            try:
                func = json.loads(out_content[i])['Functions']
                namespace = {}
                exec(func,namespace)
                active_evaluation_func = namespace['evaluation_func']
                evaluation_factors = active_evaluation_func(np.stack(local_obs, axis=0))
                factor_num = len(evaluation_factors)
                for factor in evaluation_factors:
                    if len(factor.shape) != 2 or factor.shape[0] != len(local_obs) or factor.shape[1] != 1:
                        pass_check=False
                        error_idx = i
                        error_content = f'There is an error in your previous answer. Error: The shape of the output factors should be (batch_size, 1).'
            except Exception as e:
                pass_check=False
                error_idx = i
                error_content = f'There is an error in your previous answer. Error:{e}' # with state_example: {np.round(np.stack(self.obs, axis=0), 2)}
                # break
        return pass_check, error_idx, error_content, factor_num

class MPE_simple_area_prompt(Base_prompt):
    def __init__(self, map_name='simple_area', factor_decomp=False) -> None:
        super().__init__(map_name, factor_decomp)
        from utils.util import setup_seed, make_env, n_actions
        self.env = make_env(map_name)
        self.n_others = self.env.world.n_others
        self.obs = self.env.reset()
        self.n_agents = len(self.env.world.agents)
        self.n_landmarks = len(self.env.world.landmarks)
        self.agent_size = self.env.world.agent_size
        self.collision_size = self.env.world.agent_size + self.env.world.landmarks[0].size

        self.task_description = f"TASK:We are playing Simple Area. \
In this task, {self.n_agents} agents are asked to spread out as much as possible to construct a largest triangle \
while avoiding collision(distance<={self.collision_size}) with landmarks."
        self.state_form = f"OBSERVATION FORM: At each step, each agent receives an observation array.\
This observation is represented by an array with {len(self.obs[0])} dimensions:\
concat([agent's velocity, agent's location, {self.n_others+1} nearest landmarks' relative locations, {self.n_others} nearest other agents' relative locations]).\
The agent's velocity is 2-dimensional, including x and y components.\
The agent's location is 2-dimensional, including x and y coordinates.\
The landmarks' relative locations are {self.n_others+1}x2 dimensions, including relative x and y coordinates(landmark_i.x-agent.x, landmark_i.y-agent.y).\
The other agents' relative locations are {self.n_others}x2 dimensions, including relative x and y coordinates(agent_i.x-agent.x, agent_i.y-agent.y).\n"

        self.role_instruction = factor_role_instrutor
        


def get_prompt(env_name, map_name, factor_decomp=False):
    if 'simple_spread' in map_name:
        return MPE_simple_spread_prompt(map_name, factor_decomp=factor_decomp)
    elif 'simple_tag' in map_name:
        return MPE_simple_tag_prompt(map_name, factor_decomp=factor_decomp)
    elif 'hetero_tag' in map_name:
        return MPE_hetero_tag_prompt(map_name, factor_decomp=factor_decomp)
    elif 'simple_area' in map_name:
        return MPE_simple_area_prompt(map_name, factor_decomp=factor_decomp)