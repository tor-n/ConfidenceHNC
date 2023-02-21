# Confidence-HNC (Confidence Hochbaum's Normalized Cut)
Confidence-HNC or CHNC is a graph-based semisupervised learning method that is based on Hochbaums' Normalized Cut (HNC), which is also referred to as Supervised Normalized Cut in [[1]](#1). <br>
<br/>
HNC is a graph-based method for a binary classification task where a minimum cut problem is solved on a graph instance that represents data samples as graph vertices and their similarities as edge weights. 
In addition to edges whose weights represent similarities between samples, there are also source-adjancent arcs which involve a parameter denoted as **lambda**. 
The minimum (s,t) cut of this graph provides a partition of samples that maximizes inter-similarity between two classes while minimizing intra-similarity within one of the two classes, depending on whether the edges with **lambda** weights are source-adjacent or arc-adjacent. The solution to the minimum cut problem provides the predicted labels of the unlabeled samples. For more details, see [[2]](#2). <br>
<br/>
**Lambda** is an important aspect of HNC. It is a tradeoff between the goal to minimize inter-similarity and the goal to maximize intra-similarity. Lambda is a parameter that needs to be tuned, which is done via cross validation in our work. 
While the minimum cut can be solved for each lambda value in the cross validation step, the parametric pseudoflow algorithm in [[2]](#2) can solve a sequence of minimum cut problem in the complexity of a single minimum cut problem. The instruction about the package of pseudoflow algorithm can be found [here](https://github.com/hochbaumGroup/pseudoflow-parametric-cut). <br> 
<br/>
Confidence HNC is developed from HNC with the goal to handle noisy labels. The "confidence" of the given labels of training samples or labeled samples are incorporated into the construction of the graph on which we solve for the minimum cut. <br> 
<br/>
Codes provided here are Python codes of HNC (HNC.py) and CHNC (CHNC.py). HNC has been implemented earlier but only in Matlab. Additionally, the code of HNC provided here also includes an option to use a sparsified graph as an alternative to a fully connected graph that was used in the original work of HNC. <br>
<br/>
This is a work in progress. This repository will be updated regularly.
## References
<a id="1">[1]</a> 
Baumann, Philipp, Dorit S. Hochbaum, and Yan T. Yang. "A comparative study of the leading machine learning techniques and two new optimization algorithms." European journal of operational research 272.3 (2019): 1041-1057.

<a id="2">[2]</a> 
Hochbaum, Dorit S. "The pseudoflow algorithm: A new algorithm for the maximum-flow problem." Operations research 56.4 (2008): 992-1009.
