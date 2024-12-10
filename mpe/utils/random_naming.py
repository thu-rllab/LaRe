import random

def random_naming():
    # random.seed(0)
    PASSWD_LEN = 10
    charactors_list = [chr(i) for i in range(48, 58)]
    charactors_list += [chr(i) for i in range(65, 91)]
    charactors_list += [chr(i) for i in range(97, 123)]
    pwd = ''
    for i in range(PASSWD_LEN):
        pwd += random.choice(charactors_list)
    return pwd
