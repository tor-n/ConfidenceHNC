# Confidence-HNC (Confidence Hochbaum's Normalized Cut)
Confidence-HNC or CHNC is a graph-based semisupervised learning method that is based on Hochbaums' Normalized Cut, which is also referred to as Supervised Normalized Cut in [[1]](#1). <br>
<br/>
Supervised Normalized Cut or SNC is a graph-based method for a binary classification task where a minimum cut problem is solved on a graph instance that represents data samples as graph vertices and their similarities as edge weights. The solution to the minimum cut problem provides the predicted labels of the unlabeled samples. The algorithm that is used for solving the minimum cut problem is the pseudoflow algorithm [[2]](#2). The instruction about the package of pseudoflow algorithm can be found [here](https://github.com/hochbaumGroup/pseudoflow-parametric-cut). <br> 
<br/>
The LabelConfidence-SNC or LCSNC is built upon SNC with the goal to handle noisy labels. The "confidence" of the given labels of training samples or labeled samples are incorporated into the construction of the graph on which we solve for the minimum cut. <br> 
<br/>
Codes provided here are Python codes of SNC (SNC.py) and LCSNC (LCSNC.py). SNC has been implemented earlier but only in Matlab. Additionally, the code of SNC provided here also includes an option to use a sparsified graph as an alternative to a fully connected graph that was used in the original work of SNC. <br>
<br/>
This is a work in progress. This repository will be updated regularly.
## References
<a id="1">[1]</a> 
Baumann, Philipp, Dorit S. Hochbaum, and Yan T. Yang. "A comparative study of the leading machine learning techniques and two new optimization algorithms." European journal of operational research 272.3 (2019): 1041-1057.

<a id="2">[2]</a> 
Hochbaum, Dorit S. "The pseudoflow algorithm: A new algorithm for the maximum-flow problem." Operations research 56.4 (2008): 992-1009.
