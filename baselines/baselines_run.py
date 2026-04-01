
from LinUCB import Linearucb
from SketchLinUCB import Sklinucb
from SketchLinUCBdynamic import Sklinucbdynamic
from Thompson import TS
from SketchThompson import SkTS
import argparse
import numpy as np
import sys 

from load_data import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baselines')
    parser.add_argument('--dataset', default='mnist', type=str, help='mnist, yelp, movielens, disin, synthetic')
    parser.add_argument("--method", nargs="+", default=["Neural_epsilon", "NeuralTS", "NeuralUCB", "NeuralNoExplore"], help='list: [ "LinUCB"]')
    parser.add_argument('--lamdba', default='0.1', type=float, help='Regulization Parameter')
    parser.add_argument('--nu', default='0.001', type=float, help='Exploration Parameter')
    parser.add_argument('--b_sketch',type=int,help='Sketching Dimension')
    parser.add_argument('--d',type=int,help='Feature dimension')
    parser.add_argument('--K',type=int,help='Number of arms')
    parser.add_argument('--T',type=int,help='Number of episodes')
    parser.add_argument('--csp',type=float,help='Context Sparsity',default=0.5)
    parser.add_argument('--asp',type=float,help='Action Sparsity',default=0.5)
    parser.add_argument('--data_method',type=str,help='Sparsity type',default='l2_ball')
    
    
    args = parser.parse_args()
    dataset = args.dataset
    arg_lambda = args.lamdba 
    arg_nu = args.nu
    
    d = args.d
    K = args.K
    T = args.T
    data_method = args.data_method
    action_sparsity = args.asp
    context_sparsity = args.csp
    
    print("running methods:", args.method)
    import time
    start_time = time.time()
    for method in args.method:

        regrets_all = []
        for i in range(5):
            
            # b = load_mnist_1d()
            b = load_synthetic_dataloader(data_method,d,K,T,context_sparsity,action_sparsity)
            # b = load_yelp()
            # b = load_movielen()
            
            if method == "LinUCB":
                model = Linearucb(b.dim, arg_lambda, arg_nu)
                
            elif method == "TS":
                model = TS(b.dim, arg_lambda, arg_nu)
            
            elif method == "SketchLinUCB":
                model = Sklinucb(b.dim,args.b_sketch)
                
            elif method == "SketchLinUCBdynamic":
                model = Sklinucbdynamic(b.dim,args.b_sketch)
                
            elif method == "SketchTS":
                model = SkTS(b.dim, args.b_sketch)
            else:
                print("method is not defined. --help")
                sys.exit()
                
            regrets = []
            sum_regret = 0
            print("Round; Regret; Regret/Round")
            for t in range(T):
                '''Draw input sample'''
                context, rwd = b.step()
                arm_select = model.select(context)
                reward = rwd[arm_select]

            
                model.train(context[arm_select],reward)

                regret = np.max(rwd) - reward
                sum_regret+=regret
                regrets.append(sum_regret)

                if t % 50 == 0:
                    print('{}: {:}, {:.4f}'.format(t, sum_regret, sum_regret/(t+1)))

            print("run:", i, "; ", "regret:", sum_regret)
            regrets_all.append(regrets)
            
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Program ran for {elapsed_time} seconds")
        
        if method == "SketchLinUCB" or method == "SketchTS":
            np.save(f"../results/{method}_regret_d_{d}_K_{K}_T_{T}_cs_{context_sparsity}_as_{action_sparsity}_b_{args.b_sketch}_{data_method}", regrets_all)
        else:
            np.save(f"../results/{method}_regret_d_{d}_K_{K}_T_{T}_cs_{context_sparsity}_as_{action_sparsity}_{data_method}", regrets_all)
    
    
    
    
    
    
    
    
    
    
    
    
        