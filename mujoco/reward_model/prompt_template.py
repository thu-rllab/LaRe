# from envs import REGISTRY as env_REGISTRY
import json
import numpy as np
import sys
# sys.path.append('/home/quy/LLMrd_sa')

factor_role_instrutor = f"ROLE INSTRUCTION: You are good at understanding tasks and writing python codes.\
You should fully understand the provided task and describe the exact observation and action form in the current map. \
Then, based on your understanding and the goal of the task, analyze potential positive and negative behaviours or statuses that can be reflected in the observation and action.\
Finally, write an evaluation function that returns factors evaluating the current status from different aspects. \
Note:1. Do not use information you are not given! \
2. Focus on the most relevant evaluation factors and use information in observation as little as possible. \
3. The code should be as generic, complete and not contain omissions! \
4. Avoid dividing by zero!\
5. The input variable is in the form of (batch_size, dim), please return a list of several evaluation factor arrays, each in the form of (batch_size, 1). \
Please think step by step and adhere to the following JSON format (just replace the () with your answer):"+\
"{\
Understand: (give your thought about the task),\
Analyze: (think step by step and analyze potential positive and negative behaviors or statuses that can be reflected in which part of the observation and action), \
Functions: (a python function with the form of 'def evaluation_func(observation, action):\
 ... return [a list of evaluation factor arrays]')\
}"

direct_role_instrutor = f"ROLE INSTRUCTION: You are good at understanding tasks and writing python codes.\
You should fully understand the provided task and describe the exact observation and action form in the current map. \
Then, based on your understanding and the goal of the task, analyze potential positive and negative behaviours or statuses that can be reflected in the observation and action.\
Finally, write a reward function that returns reward evaluating the current status from different aspects. \
Note:1. Do not use information you are not given! \
2. Focus on the most relevant evaluation factors and use information in observation as little as possible. \
3. The code should be as generic, complete and not contain omissions! \
4. Avoid dividing by zero!\
5. The input variable is in the form of (batch_size, dim), please return a one element list which includes reward array in the form of (batch_size, 1). \
Please think step by step and adhere to the following JSON format (just replace the () with your answer):"+\
"{\
Understand: (give your thought about the task),\
Analyze: (think step by step and analyze potential positive and negative behaviors or statuses that can be reflected in which part of the observation and action), \
Functions: (a python function with the form of 'def evaluation_func(observation, action):\
 ... return [reward]')\
}"

class Base_prompt(object):
    def __init__(self, map_name, factor=True) -> None:
        self.map_name = map_name
        self.task_description = ''
        self.state_form = ''
        self.role_instruction = ''
        self.factor = factor

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
                evaluation_factors = active_evaluation_func(np.array([self.obs]*2), np.array([self.action]*2))#, np.array([self.obs]*2))
                factor_num = len(evaluation_factors)
                if not self.factor and len(evaluation_factors) > 1:
                    pass_check=False
                    error_idx = i
                    error_content = f'There is an error in your previous answer. Error: The output should be a list with only one element, i.e., rewards.'
                for factor in evaluation_factors:
                    if len(factor.shape) != 2 or factor.shape[0] != 2 or factor.shape[1] != 1:
                        pass_check=False
                        error_idx = i
                        error_content = f'There is an error in your previous answer. Error: The shape of the output factors should be (batch_size, 1).'
            except Exception as e:
                pass_check=False
                error_idx = i
                error_content = f'There is an error in your previous answer. Error:{e.args}' # with state_example: {np.round(np.stack(self.obs, axis=0), 2)}
                # break
        return pass_check, error_idx, error_content, factor_num

    

Task_descriptions = {
    'HalfCheetah-v4': "The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws). \
The goal is to apply a torque on the joints to make the cheetah run forward (right) in x-coordinate as fast as possible, with a positive reward allocated based on the distance moved forward and a negative reward allocated for moving backward. \
The torso and head of the cheetah are fixed, and the torque can only be applied on the other 6 joints over the front and back thighs (connecting to the torso), shins (connecting to the thighs) and feet (connecting to the shins). \
Maintain safe control and prevent excessive torque norm.",
    'Walker2d-v4': " The walker is a two-dimensional two-legged figure that consist of seven main body parts - a single torso at the top (with the two legs splitting after the torso), \
two thighs in the middle below the torso, two legs in the bottom below the thighs, and two feet attached to the legs on which the entire body rests. \
The goal is to walk in the in the forward (right) direction in x-coordinate by applying torques on the six hinges connecting the seven body parts. \
Maintain safe control and prevent excessive torque norm.",
    'HumanoidStandup-v4': "The 3D bipedal robot is designed to simulate a human. It has a torso (abdomen) with a pair of legs and arms. \
The legs each consist of two links, and so the arms (representing the knees and elbows respectively). \
The environment starts with the humanoid laying on the ground. \
The goal of the environment is to make the humanoid standup and then keep it standing by applying torques on the various hinges. \
Maintain safe control and prevent excessive torque norm.",
    'Reacher-v4': "Reacher is a two-jointed robot arm. \
The goal is to move the robot's end effector (called fingertip) close to a target that is spawned at a random position. \
Maintain safe control and prevent excessive torque norm.",
}

State_forms = {
    'HalfCheetah-v4': "The observation is 18 dimensions: \
0: position: z-coordinate of the front tip; \
1: angle: angle of the front tip; \
2: angle: angle of the second rotor; \
3: angle: angle of the second rotor; \
4: angle: angle of the tip along the x-axis; \
5: angle: angle of the tip along the y-axis; \
6: angle: angular velocity of front tip; \
7: angle: angular velocity of second rotor; \
8: velocity: x-coordinate of the front tip; \
9: velocity: y-coordinate of the front tip; \
10: angular velocity: angle of the front tip; \
11: angular velocity: angle of the second rotor; \
12: angular velocity: angle of the second rotor; \
13: angular velocity: velocity of the tip along the x-axis; \
14: angular velocity: velocity of the tip along the y-axis; \
15: angular velocity: angular velocity of front tip; \
16: angular velocity: angular velocity of second rotor.\
",
    'Walker2d-v4': "The observation is 17 dimensions: \
0: position: z-coordinate of the torso (height of Walker2d); \
1: angle: angle of the torso; \
2: angle: angle of the thigh joint; \
3: angle: angle of the leg joint; \
4: angle: angle of the foot joint; \
5: angle: angle of the left thigh joint; \
6: angle: angle of the left leg joint; \
7: angle: angle of the left foot joint; \
8: velocity: velocity of the x-coordinate of the torso; \
9: velocity: velocity of the z-coordinate (height) of the torso; \
10: angular velocity: angular velocity of the angle of the torso; \
11: angular velocity: angular velocity of the thigh hinge; \
12: angular velocity: angular velocity of the leg hinge; \
13: angular velocity: angular velocity of the foot hinge; \
14: angular velocity: angular velocity of the thigh hinge (left); \
15: angular velocity: angular velocity of the leg hinge (left); \
16: angular velocity: angular velocity of the foot hinge (left).\
",
    'HumanoidStandup-v4': "The observation is 376 dimensions, the first 45 of which are about all the position and velocity: \
0: position: z-coordinate of the torso (centre); \
1: angle: x-orientation of the torso (centre); \
2: angle: y-orientation of the torso (centre); \
3: angle: z-orientation of the torso (centre); \
4: angle: w-orientation of the torso (centre); \
5: angle: z-angle of the abdomen (in lower_waist); \
6: angle: y-angle of the abdomen (in lower_waist); \
7: angle: x-angle of the abdomen (in pelvis); \
8: angle: x-coordinate of angle between pelvis and right hip (in right_thigh); \
9: angle: z-coordinate of angle between pelvis and right hip (in right_thigh); \
10: angle: y-coordinate of angle between pelvis and right hip (in right_thigh); \
11: angle: angle between right hip and the right shin (in right_knee); \
12: angle: x-coordinate of angle between pelvis and left hip (in left_thigh); \
13: angle: z-coordinate of angle between pelvis and left hip (in left_thigh); \
14: angle: y-coordinate of angle between pelvis and left hip (in left_thigh); \
15: angle: angle between left hip and the left shin (in left_knee); \
16: angle: coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm); \
17: angle: coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm); \
18: angle: angle between right upper arm and right_lower_arm; \
19: angle: coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm); \
20: angle: coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm); \
21: angle: angle between left upper arm and left_lower_arm; \
22: velocity: x-coordinate velocity of the torso (centre); \
23: velocity: y-coordinate velocity of the torso (centre); \
24: velocity: z-coordinate velocity of the torso (centre); \
25: angular velocity: x-coordinate angular velocity of the torso (centre); \
26: angular velocity: y-coordinate angular velocity of the torso (centre); \
27: angular velocity: z-coordinate angular velocity of the torso (centre); \
28: angular velocity: z-coordinate of angular velocity of the abdomen (in lower_waist); \
29: angular velocity: y-coordinate of angular velocity of the abdomen (in lower_waist); \
30: angular velocity: x-coordinate of angular velocity of the abdomen (in pelvis); \
31: angular velocity: x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh); \
32: angular velocity: z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh); \
33: angular velocity: y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh); \
34: angular velocity: angular velocity of the angle between right hip and the right shin (in right_knee); \
35: angular velocity: x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh); \
36: angular velocity: z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh); \
37: angular velocity: y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh); \
38: angular velocity: angular velocity of the angle between left hip and the left shin (in left_knee); \
39: angular velocity: coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm); \
40: angular velocity: coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm); \
41: angular velocity: angular velocity of the angle between right upper arm and right_lower_arm; \
42: angular velocity: coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm); \
43: angular velocity: coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm); \
44: angular velocity: angular velocity of the angle between left upper arm and left_lower_arm.\
",
    'Reacher-v4': "The observation is 11 dimensions: \
0: unitless: cosine of the angle of the first arm; \
1: unitless: cosine of the angle of the second arm; \
2: unitless: sine of the angle of the first arm; \
3: unitless: sine of the angle of the second arm; \
4: position: x-coordinate of the target; \
5: position: y-coordinate of the target; \
6: angular velocity: angular velocity of the first arm; \
7: angular velocity: angular velocity of the second arm; \
8: position: x-value of position_fingertip - position_target; \
9: position: y-value of position_fingertip - position_target; \
10: position: z-value of position_fingertip - position_target (constantly 0 since reacher is 2d and z is same for both).\
",
}

Action_forms = {
    'HalfCheetah-v4': "The action is 6 dimensions. An action represents the torques applied at the hinge joints.",
    'Walker2d-v4': "The action is 6 dimensions. An action represents the torques applied at the hinge joints.",
    'HumanoidStandup-v4': "The action is 17 dimensions. An action represents the torques applied at the hinge joints.",
    'Reacher-v4': "The action is 2 dimensions. An action represents the torques applied at the hinge joints.",
    }

    
class Mujoco_prompt(Base_prompt):
    def __init__(self, map_name='HalfCheetah-v4', factor=True) -> None:
        super().__init__(map_name)
        import gymnasium as gym
        self.env = gym.make(map_name)#, exclude_current_positions_from_observation=False)
        self.obs, _ = self.env.reset()
        self.action = self.env.action_space.sample()
        self.task_description = f"TASK: {Task_descriptions[map_name]}\n"
        self.state_form = f"OBSERVATION FORM: {State_forms[map_name]}\n"
        self.action_form = f"ACTION FORM: {Action_forms[map_name]}\n"
        self.role_instruction = factor_role_instrutor if factor else direct_role_instrutor