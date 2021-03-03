# Simplifying Architecture Search for Graph Neural Network

#### Overview
To obtain optimal data-specific GNN architectures, we propose the SNAG framework (Simplified Neural Architecture search for Graph neural
networks), consisting of a simpler yet more expressive search space and a RL-based search algorithm.

We conduct our experiments based on [GraphNAS](https://github.com/GraphNAS/GraphNAS) and [PyG](https://github.com/rusty1s/pytorch_geometric). The main difference compared with GraphNAS:

    1. We provide the implementation of weight sharing strategy.
    2. The finetuning stage of GNN.
    3. The implementations of Random and Bayesian search algorithms.
    
#### Requirements
Latest version of Pytorch-geometric(PyG) is required. More details can be found in [here](https://github.com/rusty1s/pytorch_geometric)
#### Architecture Search
Search a 3-layer GNN on Cora dataset based on the designed search space, please run:
    
    (change folder name......)
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --shared_initial_step 10   --shared_params True  #SNAG-WS
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  #SNAG


Other NAS methods, e.g., Random, Bayesian and GraphNAS in Section 4, please run:
    
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --search_mode graphnas    #GraphNAS
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --search_mode graphnas  --shared_initial_step 10   --shared_params True  #GraphNAS-WS
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --mode random #Random
    python -m rlctr.main  --dataset Cora   --layers_of_child_model 3  --mode bayes  #Bayesian

#### Cite
Please cite [our paper](https://arxiv.org/pdf/2008.11652.pdf) if you use this code:

    @Technicalreport{zhao2020simplifying,
    title={Simplifying Architecture Search for Graph Neural Network},
    author={Zhao, Huan and Wei, Lanning and Yao, Quanming},
    journal={arXiv preprint arXiv:2008.11652},
    year={2020}
    }

