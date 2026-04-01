import numpy as np
import matplotlib.pyplot as plt


def get_mean_std(ress):
    return np.mean(ress, axis=0), np.std(ress, axis =0)
    

if __name__ == '__main__':    
    T = 10000
    x = range(T)
    plt.figure(figsize=(10, 6))

    ucb = np.load("results_new/LinUCB_regret_d_500_K_10_T_2000_cs_0.0_as_0.9_random_ball.npy")
    ucb = ucb[:,:2000]
    ucb_mean, ucb_std = get_mean_std(ucb)
    plt.plot(range(2000), ucb_mean, 'k-', color='blue',linewidth=2.0,linestyle=':', label = 'LinUCB')
    plt.fill_between(range(2000), ucb_mean-ucb_std, ucb_mean+ucb_std, facecolor='blue', alpha=0.2)


    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.legend()
    plt.title("Mnist")
    plt.savefig('./figures/regret_mnist.pdf', dpi=500)