# Neural Algorithmic Reasoning Index

A proto-index, perhaps, abiding by today's zeitgeist, a sort of worklog. An attempt to gain proficiency in the more practical aspects of graph representation learning. In a sense, reconciling too much paper reading with something more visceral. Implemented in MLX and at times accompanied by commentary, the techniques/methods/papers implemented were curated by myself and, as such, suffer from the noble curse of at times seeming too scattered. An attempt was made to nonetheless keep them focused around the area of Neural Algorithmic Reasoning, hence the name. *Work in progress*

| Architecture | Results |
| --- | --- |
| Neural Turing Machines | Unsuccessful |
| Graph Attention Networks | 83% acc. (CORA) |
| Message Passing Neural Networks | 80% acc (CORA) |
| Neural Execution Networks | **Training** |
| Generalist Neural Algorithmic Learner | *Backlog* |
| Graph Variational Autoencoder | *Backlog* |
|Counterfactual G-invariance regularization| *Backlog* |


### Usage
```bash
# Graph Attention Networks (GAT) on CORA dataset
cd implementations/gat
python experiment.py --total-steps 2000 --learning-rate 0.003

# Message Passing Neural Networks (MPNN) 
cd implementations/mpgnn
python experiment.py --aggregation max --total-steps 2000
```

All experiments include detailed parameter options via `--help`.


### Neural Turing Machines
More than anything, I am now convinced that Neural Turing Machines (NTMs) reflect the will of their creators to somehow manage the ungodly gradient dynamics that emerge during training—especially in the feedforward variant. This challenge is made worse not only by the scarcity of open-source implementations but, more surprisingly, by the near absence of substantial literature investigating them. Many hours were spent, nonetheless, modifying the architecture and training loop in hopes of replicating the original paper’s findings. This aim, however, was not achieved, partly due to serendipitous gradient collapses to NaN, and partly due to insufficient access to compute resources. Intuitively, it seems that the tasks we expect MLPs to perform under this paradigm lie right at the boundary of their representational capacity. Or perhaps more accurately, these are tasks for which the inductive biases inherent in traditional MLP architectures are fundamentally ill-suited, making their training a maximally sample-complex regime. It does feel odd, though, that so little is said about them, particularly given how intuitive they seem as a natural extension of regular feedforward networks.

Implementation-wise, the only notable deviation from what might be called a “standard” approach—if such a standard exists—is the adoption of a branching architecture in which the controller and output branches are isolated from one another while sharing a common preprocessing module. That said, I have abstained from providing a detailed architectural description, since most of it can be inferred from the code, it failed to achieve any notable result, and lastly because LLMs render any attempt at a technical report somewhat redundant.

### Graph Attention Networks & MPNN
Naively, I never put much thought into learning PyTorch (same goes for MLX). That is, I treated it simply as a sort of translation process—one where semantics were crucial and syntax merely an afterthought. This proved to be a costly mistake. Many hours and many burned laps were needed in order to make me realize that I couldn’t simply loop my way through life and expect that blindness to vectorization, parallelization, etc., wouldn’t eventually come back to bite. Now, although I must admit it is not the most pleasant of activities, there is something satisfying in finally nailing the tensor acrobatics needed to make the models run efficiently. More precisely, I now see some beauty in neighborhood-aggregated softmax, in a way I was blind to months before. Here I must say Aleksa Gordić’s[notebooks](https://github.com/gordicaleksa/pytorch-GAT?tab=readme-ov-file) were invaluable.



  