# trust_connectomes
Code for pre-print "Can we trust machine learning in fMRI? Simple adversarial attacks break connectome-based predictive models"

## adversarial_noise
adversarial noise attacks on two-class classification
functional connectomes are altered to cause misclassification
use run_adv_noise.m to run iterations of this

## enhancement
enhancement attacks, where data is perturbed to *falsely* improve performance
use enhancement_attack.m to run iterations of this

## task_swap
swapping the task at test time (HCP data) to alter performance
use run_trust_taskswap.m to run iterations of this

## utils
Please add this to your path. Also download and add to your path the Brain Connectivity Toolbox (https://sites.google.com/site/bctnet/)

## References
Brain Connectivity Toolbox: https://sites.google.com/site/bctnet/ (download this code for analysis)
Yale MRRC Github: https://github.com/YaleMRRC/CPM (acted as base for some code in this project)
Biggio et al. 2013: https://link.springer.com/content/pdf/10.1007/978-3-642-40994-3_25.pdf (attack method)

Feel free to e-mail matthew.rosenblatt@yale.edu with any questions or concerns
