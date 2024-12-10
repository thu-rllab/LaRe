from re import A
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os

def plot_test_reward(csv):
    csv = pd.DataFrame(csv.values.T, columns=csv.index, index=csv.columns).reset_index()
    csv = pd.DataFrame(csv.iloc[1:].values, columns=csv.iloc[0])
    
    x = np.array(csv['steps']).astype(np.float)/(50*1000)
    y = np.around(np.array(csv['rewards']).astype(np.float), 4)
    x_sticks = np.arange(0, 200, 20)
    y_sticks = np.arange(-100, 100, 50)

    plt.xlabel('Episodes * 1000')
    plt.ylabel('Agent Reward')
    plt.plot(x, y, linewidth=1.5)
    plt.xticks(x_sticks)
    plt.yticks(y_sticks)
    plt.savefig('./images/Mean Reward')

# records = pd.read_csv('./results/log/simple_tag_n3_STAS_w_vn/train_curve.csv')
# plot_test_reward(records)


# def plot_reward(records):
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    
    # # Make data.
    # Epi, T = records.shape
    # X = np.arange(0, Epi, 1)
    # Y = np.arange(0, T, 1)
    # # X, Y = np.meshgrid(X, Y)
    # # Z = np.array(records).T
    # # print(X, Y)
    # # R = np.sqrt(X ** 2 + Y ** 2)
    # # Z = np.sin(R)
    # # print(Z.shape, type(Z), Z)
    # # print(vev)
    
    # # Plot the surface.
    # # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    # #                     linewidth=0, antialiased=False)
    # for i in range(Epi):
    #     ax.plot(Y, records[i], zs=i, zdir='y', marker='o', alpha=0.8)
    
    # xticks = np.arange(0, T, 5)
    # yticks = np.arange(0, Epi, 1)
    # plt.xticks(xticks)
    # plt.yticks(yticks)
    # # Customize the z axis.
    # ax.set_zlim(records.min(), records.max())
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # # Add a color bar which maps values to colors.
    # # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig('./images/reward_decomposition')
    
    # # plt.show()
def plot_reward(decomposed_r, real_r, info, i_epi, logdir, plot):
    if plot:
        plt.figure(dpi=80)
        plot = False
    else: pass
    plot_time_reward(decomposed_r.sum(0), real_r.sum(0), info.sum(0), i_epi, logdir)
    plot_agent_reward(decomposed_r, real_r, info, i_epi, logdir)
    return plot

def plot_agent_reward(decomposed_r, real_r, info, i_episode, logdir):
    r_good = np.sum(decomposed_r*(info>0), axis=0)
    r_bad = np.sum(decomposed_r*(info<1), axis=0)

    x = np.arange(decomposed_r.shape[1])
    total_width, n = 0.5, decomposed_r.shape[0]
    width = total_width / n
    x = x - (total_width - width) / 1

    info = info.sum(0)

    plt.plot(x, r_good, color='cadetblue', marker='o', label='Good Coalition', alpha=0.8)
    plt.plot(x, r_bad, color='red', marker='x', label='Bad Coalition', alpha=0.8)
    for i in np.arange(decomposed_r.shape[1]):
        if info[i]>0:
            # print(i, info[i], real_r[i])
            plt.axvline(x[i], color='#4A708B', linestyle='--',linewidth=1, alpha=0.8)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.title("Reward Decomposition for Agent Along Time Step",color='#4A708B',fontsize=14)
    plt.savefig(os.path.join(logdir, 'Return_Decomposition_in_Agent_episode_{}.png'.format(i_episode)))
    plt.clf()

def plot_time_reward(decomposed_r, real_r, info, i_episode, logdir):
    x = np.arange(decomposed_r.shape[0])

    plt.style.use('ggplot')
    plt.plot(x, decomposed_r, linewidth=2, color = 'red', label='decomposed', alpha=0.7)
    plt.plot(x, real_r, linewidth=2, color='#3282B8', label='real', alpha = 0.7)
    for i in x:
        if info[i]>0:
            # print(i, info[i], real_r[i])
            plt.axvline(i, color='#4A708B', linestyle='--',linewidth=1, alpha=0.8)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.title("Reward Decomposition Along Time Step",color='#4A708B',fontsize=14)
    plt.savefig(os.path.join(logdir, 'Return_Decomposition_in_Time_episode_{}.png'.format(i_episode)))
    plt.clf()
    



# test_data = np.random.randn(10,25)
# plot_agent_reward(np.random.randn(3,25), np.random.randn(3,25), np.random.randint(0,2,size=(3, 25)), 0, './')