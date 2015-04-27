import numpy as np

def mv_rejective(models, n_samples=1e3):
    '''
    Multivariate rejective sampling method
    --------------------------------------
    This function receives histograms and edges of the histogram as a model 
    to compute new samples with the same histograms.
    
    Input
    -----
    models : list of models that include normalized histogram and edges
    n_samples: number of samples to generate
    
    Output
    ------
    new_samples : n_samples R.V. samples of len(models) dimension
    '''
    
    n_accepted = 0
    MUL = 0
    # While the number of accepted samples is not enough
    while n_accepted < n_samples:
        new_samples= []
        # multiplier that determine the number of candidate samples to compute
        MUL = MUL + 15
        # For each model
        for h_count, edges in models:
            # normalize histogram bin area
            h_count = h_count * np.diff(edges)
            # es ~ U(min,max)
            es = np.random.uniform(low = edges[0],
                                   high = edges[-1] ,
                                   size = n_samples * MUL)
            # us ~ U(0,1)
            us = np.random.uniform(size = n_samples * MUL)
            # Keep accepted samples
            column = [e for e,u in zip(es,us) if u <= h[(e>edges[1:]).argmin()]]
            new_samples.append(column)
        # Number of accepted multivariate samples 
        n_accepted = min([len(a) for a in new_samples])

    # convert the list into numpy array of size Nsample by Number of
    # components
    new_samples = np.array([column[0:Nsample] for column in new_samples]).T
    return(new_samples)

