# LabelConfidence-SNC (Supervised Normalized Cut)
LabelConfidence-SNC is a graph-based semisupervised learning method that is based on Supervised Normalized Cut (Baumann, Hochbauma and Yang, [2019] (https://www.sciencedirect.com/science/article/pii/S0377221718306143?casa_token=59CAaNpPp2sAAAAA:R1O0XejpAsTxMdUHprUkMekuccsckUwZjfAfyUN4Wv7qPiInrV533ZoipmqreTADjrbPS-Cw7PQ)). <br>
<br/>
Supervised Normalized Cut or SNC is a graph-based method for a binary classification task where a minimum cut problem is solved on a graph instance that represents data samples as graph vertices and their similarities as edge weights. The solution to the minimum cut problem provides the predicted labels of the unlabeled samples. The algorithm that is used for solving the minimum cut problem is the pseudoflow algorithm (Hochbaum, [2008] (https://pubsonline.informs.org/doi/abs/10.1287/opre.1080.0524)). <br> 
<br/>
The LabelConfidence-SNC or LCSNC is built upon SNC with the goal to handle noisy labels. The "confidence" of the given labels of training samples or labeled samples are incorporated into the construction of the graph on which we solve for the minimum cut. 
Codes provided here are Python codes of SNC and LCSNC. SNC has been implemented earlier but only in Matlab. 
