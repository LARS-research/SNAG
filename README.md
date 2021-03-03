# Simplifying Architecture Search for Graph Neural Network

#### Overview
This is the code for our paper [Simplified Neural Architecture search for Graph Neural Networks](https://arxiv.org/pdf/2008.11652.pdf), publised in CSSA-CIKM 2020.
It is a neural architecture search (NAS) for graph neural network (GNN).
To obtain optimal data-specific GNN architectures, we propose the SNAG framework, consisting of a simpler yet more expressive search space and a RL-based search algorithm.

The framewwork is implemented on top of [GraphNAS](https://github.com/GraphNAS/GraphNAS) and [PyG](https://github.com/rusty1s/pytorch_geometric). The main difference compared with GraphNAS:

    1. We provide the implementation of weight sharing strategy.
    2. The finetuning stage of GNN.
    3. The implementations of Random and Bayesian search algorithms.
    
#### Requirements
Latest version of Pytorch-geometric(PyG) is required. More details can be found in [here](https://github.com/rusty1s/pytorch_geometric)

    Python == 3.7.4   Pytorch-geometric>=1.6.3   PyTorch == 1.6.0 

#### Architecture Search
Search a 3-layer GNN on Cora dataset based on the designed search space, please run:
    
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --shared_initial_step 10   --shared_params True  #SNAG-WS
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  #SNAG


Other NAS methods, e.g., Random, Bayesian and GraphNAS in Section 4, please run:
    
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --search_mode graphnas    #GraphNAS
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --search_mode graphnas  --shared_initial_step 10   --shared_params True  #GraphNAS-WS
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --mode random #Random
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --mode bayes  #Bayesian

#### Cite
Please kindly cite [our paper](https://arxiv.org/pdf/2008.11652.pdf) if you use this code:

    @Technicalreport{zhao2020simplifying,
    title={Simplifying Architecture Search for Graph Neural Network},
    author={Zhao, Huan and Wei, Lanning and Yao, Quanming},
    journal={arXiv preprint arXiv:2008.11652},
    year={2020}
    }
    
#### Misc
If you have any questions about this project, you can open issues, thus it can help more people who are interested in this project. We will reply to your issues as soon as possible. You are also welcomed to reach us by weilanning@4paradigm.com
