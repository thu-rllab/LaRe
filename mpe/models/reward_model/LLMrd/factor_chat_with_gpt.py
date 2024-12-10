import requests
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompt_template import *
import numpy as np
import argparse
import torch as th
from openai import OpenAI
global_port = 8001

def inference(model, message, temperature,host, n=1, seed=None):
    data=json.dumps({'model':model,'messages':message, 'temperature':temperature,'response_format':{ "type": "json_object" }, 'n':n})
    correct_inference = False
    count = 0
    out_content = []
    while not correct_inference:
        count += 1
        if count >= 5:
            break
        try:
            client = OpenAI()
            out = client.chat.completions.create(model=model, messages=message, temperature=temperature, response_format='json_object', n=n)
            # out = requests.post(host,data=data)
            
            for i in range(n):
                out_content.append(json.loads(out.text)['choices'][i]['message']['content'])
            correct_inference = True
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f'Error: {e}. Retrying...')
    return out_content


def callgpt(env_name, map_name, save_dir,save=True, id=0, use_recheck=False, n=5, port=8000, seed=0):
    if port is None:
        port = global_port
    prompt = get_prompt(env_name, map_name, factor_decomp=True)
    message = prompt.get_message()
    all_message = prompt.get_message()
    start_idx = len(all_message)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    recheck_message = [
        {'role':'user','content':"You have generated several evaluation functions. \
Please summarize them and generate a new evaluation function that incorporates all the evaluation factors.\
If there are other important evaluation factors, please include them as well."},
        ]
    if n==1:
        recheck_message = []

    model='gpt-4o'
    init_temperature = 1.0
    check_temperature = 0.3
    host =  None
    out_content = inference(model, message, init_temperature, host, n=n, seed=seed)
    message.append({'role':'assistant','content':str(out_content)})
    all_message.append({'role':'assistant','content':str(out_content)})
    print(out_content)
    check_phases = len(recheck_message) if len(recheck_message) > 0 else 1


    for i in range(check_phases):
        if len(recheck_message) > 0:
            new_content = recheck_message.pop(0)['content']
            # while True:
            message.append({'role':'user','content':new_content})
            all_message.append({'role':'user','content':new_content})
            out_content = inference(model, message, check_temperature, host,  seed=seed)
            print(out_content)

            # message[start_idx] = {'role':'assistant','content':str(out_content)}
            message.append({'role':'assistant','content':str(out_content)})
            # message = message[:start_idx+1]
            all_message.append({'role':'assistant','content':str(out_content)})

        for recheck_count in range(5):
            pass_check, error_idx, error_content, factor_num = prompt.factor_check(out_content)
            if pass_check:
                break
            if recheck_count == 0:
                message[-1] = {'role':'assistant','content':out_content[error_idx]}
                message.append({'role':'user','content':error_content})
                all_message.append({'role':'user','content':error_content})
            else:
                message[-2] = {'role':'assistant','content':out_content[error_idx]}
                message[-1] = {'role':'user','content':error_content}
                all_message.append({'role':'assistant','content':out_content[error_idx]})
                all_message.append({'role':'user','content':error_content})
            out_content = inference(model, message, check_temperature, host, seed=seed)
            all_message.append({'role':'assistant','content':str(out_content)})
            print(out_content)

        # message[start_idx] = {'role':'assistant','content':str(out_content)}
        # message = message[:start_idx+1]
        

        if save and pass_check:
            np.save(save_dir+f'/response_{id}.npy', out_content)
            np.save(save_dir+f'/dialog_{id}.npy', all_message)
            np.save(save_dir+f'/factor_num_{id}.npy', factor_num)
        # break


def hetero_callgpt(env_name, map_name, save_dir,save=True, id=0, use_recheck=False, n=0, port=None, seed=None):
    if port is None:
        port = global_port
    prompt = get_prompt(env_name, map_name, factor_decomp=True)
    for hetero_id in range(2):
        message = prompt.get_message()[hetero_id]
        all_message = prompt.get_message()[hetero_id]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        recheck_message = [
            {'role':'user','content':"You have generated several evaluation functions. \
    Please summarize them and generate a new evaluation function that incorporates all the evaluation factors.\
    If there are other important evaluation factors, please include them as well."},]

        model='gpt-4o'
        init_temperature = 1.0
        check_temperature = 0.3
        host =  None
        out_content = inference(model, message, init_temperature, host, n=n, seed=seed)
        message.append({'role':'assistant','content':str(out_content)})
        all_message.append({'role':'assistant','content':str(out_content)})
        print(out_content)
        check_phases = len(recheck_message)


        for i in range(check_phases):
            new_content = recheck_message.pop(0)['content']
            # while True:
            message.append({'role':'user','content':new_content})
            all_message.append({'role':'user','content':new_content})
            out_content = inference(model, message, check_temperature, host,  seed=seed)
            print(out_content)

            message.append({'role':'assistant','content':str(out_content)})
            all_message.append({'role':'assistant','content':str(out_content)})

            for recheck_count in range(5):
                pass_check, error_idx, error_content, factor_num = prompt.factor_check(out_content, hetero_id)
                if pass_check:
                    break
                if recheck_count == 0:
                    message[-1] = {'role':'assistant','content':out_content[error_idx]}
                    message.append({'role':'user','content':error_content})
                    all_message.append({'role':'user','content':error_content})
                else:
                    message[-2] = {'role':'assistant','content':out_content[error_idx]}
                    message[-1] = {'role':'user','content':error_content}
                    all_message.append({'role':'assistant','content':out_content[error_idx]})
                    all_message.append({'role':'user','content':error_content})
                out_content = inference(model, message, check_temperature, host, seed=seed)
                all_message.append({'role':'assistant','content':str(out_content)})
                print(out_content)

            if save and pass_check:
                np.save(save_dir+f'/response_{id}_{hetero_id}.npy', out_content)
                np.save(save_dir+f'/dialog_{id}_{hetero_id}.npy', all_message)
                np.save(save_dir+f'/factor_num_{id}_{hetero_id}.npy', factor_num)
    return prompt
