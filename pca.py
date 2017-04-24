from numpy import *
import matplotlib.pyplot as plt

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    redEigVal = eigVals[eigValInd] # return reduced eig val for whitening
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    lowDDataMat = lowDDataMat / redEigVal**0.5 # whiten data
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, redEigVects, meanVals, redEigVal

def plotBestFit(dataSet1,dataSet2):      
    dataArr1 = array(dataSet1)
    dataArr2 = array(dataSet2)
    n = shape(dataArr1)[0] 
    n1=shape(dataArr2)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    xcord3=[];ycord3=[]
    j=0
    for i in range(n):
        xcord1.append(dataArr1[i,0]); ycord1.append(dataArr1[i,1])
        xcord2.append(dataArr2[i,0]); ycord2.append(dataArr2[i,1])
                   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()    