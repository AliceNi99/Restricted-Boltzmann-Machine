import numpy as np

class rbm:

    def __init__(self, n_visible, n_hidden):
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        
        #As Hinton suggests, we use small random values for the weights chosen 
        #from a zero-mean Gaussian with a standard deviation of 0.01.
        self.W=np.random.normal(0, 0.1, size=(n_visible,n_hidden) )
        self.h_bias=np.zeros(n_hidden)
        self.v_bias=np.zeros(n_visible)

        

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
        
    def encoding(self,data,sampling):
        self.v1=data
        tmp=self.h_bias + (self.v1.T @ self.W)
        self.h1_distr=self.sigmoid(tmp) #distribution of h given v
        if sampling==False:
            return self.h1_distr
        else:
            self.h1=self.h1_distr > np.random.rand(self.n_hidden) #sampling of h
            return self.h1

    def reconstruction(self,sampling):
        tmp=self.v_bias+(self.W @ self.h1)
        self.v2_distr=self.sigmoid(tmp) #distribution of v given h
        if sampling==False:
            return self.v2_distr
        else:
            self.v2=self.v2_distr>np.random.rand(self.n_visible) #sampling of reconstructed v
            return self.v2
    
    def fit(self, data, lr, epochs):
        errors=np.zeros(data.shape[0])
        epoch_errors=np.zeros(epochs)
        for ep in range(epochs):
            for i in range(data.shape[0]):    
                self.encoding(data[i,:],sampling=True)
                self.reconstruction(sampling=True)
                wake=np.outer(self.v1,  self.h1_distr)
                tmp=self.h_bias+ (self.v2.T @ self.W)
                self.h2_distr=self.sigmoid(tmp)
                dream=np.outer(self.v2, self.h2_distr)               
                #update weights matrix and bias vectors according to contrastive divergence
                self.W = self.W + lr*(wake-dream)
                self.h_bias=self.h_bias + lr*(self.h1_distr-self.h2_distr)
                self.v_bias=self.v_bias + lr*(self.v1-self.v2) 
                #error for each training example
                error=np.mean(np.abs(self.v1-self.v2))
                errors[i]=error
            #vector of the errors at each epoch
            epoch_errors[ep]=np.mean(errors)
            print('error=',epoch_errors[ep], 'at epoch=',ep,'\n')
        return epoch_errors