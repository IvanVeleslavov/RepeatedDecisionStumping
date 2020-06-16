# Repeated Decision Stumping (ReDX)

ReDX is a feature selection approach utilising decision stumps models to rank and evaluate features using information gain. We apply this method to single-cell gene expression data of murine neural development and _Xenopus tropicalis_ embryogenesis and evaluate the performance of the learnt models. Repository provided in support of pre-published manuscript "Repeated Decision Stumping Distils Simple Rules from Complex Data" by Ivan Croydon-Veleslavov and Professor Michael Stumpf.


### Contents
Core functionality provided in the Julia module [RepeatedDecisionStumping.jl](code/RepeatedDecisionStumping.jl). A Jupyter notebook [ReDX_notebook.ipynb](code/ReDX_notebook.ipynb) is also provided for regenerating the results contained in the paper, as well as a Julia script [S1.jl](code/S1.jl) for generating the supplementary results.

