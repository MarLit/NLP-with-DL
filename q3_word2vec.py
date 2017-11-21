#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    
    x = x/np.sqrt((x * x).sum(axis = 1, keepdims = True))#norm
    
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print ("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    
    # Note: numpy does not feel any difference between 1D row and column vectors! 
    # Therefore, we can omit:
    # U = outputVectors.T to make output vectors be columns, as in the assignment
    
    V, d = outputVectors.shape     #u0 = outputVectors[target,:] 
    y_hot = np.zeros(V)
    np.put(y_hot, [target],[1])
    y_hat = softmax(np.dot(outputVectors, predicted))
    cost = - np.log(y_hat[target])
    gradPred = np.dot(outputVectors.T,(y_hat - y_hot))     
    grad = (y_hat - y_hot).reshape(V,1)*predicted.reshape(1,d)
    
    ################################################################################
    
    #scores = softmax(outputVectors.dot(predicted).reshape(1, V)).reshape(V,)  # same as my y_hat
    #labels = np.zeros(V)
    #labels[target] = 1          # same as y_hot
    #dscores = scores - labels   # same as (y_hat - y_hot)
    #cost1 = - np.log(scores[target])
    #gradPred1 = dscores.dot(outputVectors)
    #grad1 = dscores.reshape(V, 1).dot(predicted.reshape(d, 1).T) 
    
    ################################################################################
          
    #if np.allclose(cost1, cost, 1e-05) == False: print("!!! problem in COST in softmaxCostAndGradient !!!")      
    #if np.allclose(gradPred1, gradPred, 1e-05) == False: print("!!! problem in GRAD_PRED in softmaxCostAndGradient !!!") 
    #if np.allclose(grad1, grad, 1e-05) == False: print("!!! problem in GRAD in softmaxCostAndGradient !!!") 
    #else : print("softmaxCostAndGradient is fine...")
    
    #print("checkpoint 1:", cost1 - cost)
    #print("checkpoint 2:", gradPred1 - gradPred)
    #1print("checkpoint 3:", grad1 - grad)
 
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K): #range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K)) # [target_index, contex_index_1, ... contex_index_K]
    
    ### YOUR CODE HERE

    V, d = outputVectors.shape
    u0 = outputVectors[target,:] 
    cost = -np.log(sigmoid(np.dot(u0,predicted)))
    gradPred = -(1-sigmoid(np.dot(u0,predicted)))*u0 
    grad = np.zeros((V,d))
    grad[target,:] = -(1-sigmoid(np.dot(u0,predicted)))*predicted
        
    for k in xrange(1, len(indices)):
        index = indices[k]
        uk = outputVectors[index,:]
        cost = cost - np.log(sigmoid(np.dot(-uk,predicted)))
        gradPred = gradPred + (1-sigmoid(-np.dot(uk,predicted)))*uk
        grad[index,:]= grad[index,:]+((1-sigmoid(-np.dot(uk,predicted)))*predicted)
    
    ################################################################################
    
    #V, D = outputVectors.shape
    
    #sampleIndexs = indices #otherwise his indiced differ from implemented in the assignment
    #del sampleIndexs[0]
    ###for i in range(K):
    ####    index = dataset.sampleTokenIdx()
    ####    sampleIndexs.append(index)
    #sampleVectors = outputVectors[sampleIndexs, :]
    
    #w_r_out = sigmoid(outputVectors[target].dot(predicted))
    #w_r_k = sigmoid(-sampleVectors.dot(predicted))
    
    #cost1 = - np.log(w_r_out) - np.sum(np.log(w_r_k))  
    #gradPred1 = outputVectors[target] * (w_r_out - 1) +  (1 - w_r_k).dot(sampleVectors)
    #grad1 = np.zeros(outputVectors.shape)
    
    #grad1[target] = predicted * (w_r_out - 1)
    #for i in range(K):
    #    grad1[sampleIndexs[i]] += predicted * (1 - w_r_k)[i] 
    
    ################################################################################
    
    #if np.allclose(cost1, cost, 1e-05) == False: print("!!! problem in COST in negSamplingCostAndGradient !!!")      
    #if np.allclose(gradPred1, gradPred, 1e-05) == False: print("!!! problem in GRAD_PRED in negSamplingCostAndGradient !!!") 
    #if np.allclose(grad1, grad, 1e-05) == False: print("!!! problem in GRAD in negSamplingCostAndGradient !!!") 
    #else : print("negSamplingCostAndGradient is fine...")
    
    #print("checkpoint 1:", cost1 - cost)
    #print("checkpoint 2:", gradPred1 - gradPred)
    #print("checkpoint 3:", grad1 - grad)
    
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    
    ### YOUR CODE HERE
    
    V, N = inputVectors.shape
    
    vc_number = tokens[currentWord]
    vc = inputVectors[vc_number]
            
    for k in xrange(len(contextWords)):
        index = tokens[contextWords[k]]
        #print("k =", k, "index =", index)
        cost_k, gradPred_k, grad_k = word2vecCostAndGradient(vc, index, outputVectors, dataset)
        #print("...:", cost_k, gradPred_k, grad_k)
        cost = cost + cost_k
        gradIn[vc_number] = gradIn[vc_number] + gradPred_k
        gradOut = gradOut + grad_k
        
       
    ################################################################################################
      
    #cost1 = 0.0
    #gradIn1 = np.zeros(inputVectors.shape)
    #gradOut1 = np.zeros(outputVectors.shape)
    #V, N = inputVectors.shape
    
    #currentIndex = tokens[currentWord]
    #currentVector = inputVectors[currentIndex]     # same as my vc
    
    ##softmaxCostAndGradient(predicted, target, outputVectors)
    #for context_word in contextWords:
    
    #    target = tokens[context_word]
    #    #print("index =", target)
    #    curr_cost, curr_grad_in, curr_grad_out = word2vecCostAndGradient(currentVector, target, outputVectors, dataset)
    #    #print("...:", curr_cost, curr_grad_in, curr_grad_out)
    #    cost1 += curr_cost
    #    gradIn1[currentIndex] += curr_grad_in
    #    gradOut1 += curr_grad_out    
    
    ################################################################################################
    
    #if np.allclose(cost1, cost, 1e-05) == False: print("!!! problem in COST in skipgram !!!")      
    #if np.allclose(gradIn1, gradIn, 1e-05) == False: print("!!! problem in GRAD_IN in skipgram !!!") 
    #if np.allclose(gradOut1, gradOut, 1e-05) == False: print("!!! problem in GRAD_OUT in skipgram !!!") 
    #else : print("skipgram is fine...")
    
    #print("checkpoint 1:", cost1 - cost)
    #print("checkpoint 2:", gradIn1 - gradIn)
    #print("checkpoint 3:", gradOut1 - gradOut)
    
    ''' 
    - checkpoints work perfectly well for sigmoidCostAndGradient, and do not work for negSamplingCostAndGradient -- OK
    - even for sigmoidCostAndGradient, it says that gradient-check fails...
    '''
  
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):#range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]#xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print ("==== Gradient check for skip-gram ====")
    print("___softmaxCostAndGradient___") #####
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    print("___negSamplingCostAndGradient___") #####
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print ("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print ("\n=== Results ===")
    print (skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))
    print (cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()