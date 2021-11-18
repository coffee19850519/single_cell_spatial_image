import numpy as np
import matplotlib.pyplot as plt
import os

# Generate simulation
# We generate different simulation models, try to check whether simulation can help


# Model1:
# 1222221
# 1222221
# ...
# 1222221
#
# Model2:
# 11112221
# 11122211
# ...
# 12221111
#
# Model3:
# 12222221
# 11122111
# 11221111
# ...
# 12222211
#
# Model4:
# 12222221
# 12122111
# 12112221
# ...
# 12212221
#
# Model5:
# 1222223
# 1222223
# ...
# 1222223
#
# Model6:
# 11112223
# 11122233
# ...
# 12223333
#
# Model7:
# 12222223
# 11122333
# 11223333
# ...
# 12222233
#
# Model8:
# 12222223
# 12122333
# 12112223
# ...
# 12212223
def selectModel(modelName,labelArray):
    outLines=''
    tmpLine='barcode'
    for i in range(locSize*locSize):
        tmpLine=tmpLine+',cell'+str(i)
    outLines=outLines+tmpLine+'\n'

    for i in range(2000):
        tmpLine='g'+str(i)
        if modelName == 'model1' or modelName == 'model2' or modelName == 'model3' or modelName == 'model4':
            if i<1000:
                for j in range(totalSize):
                    if labelArray[j]==1:
                        ex = 0.0+np.random.rand()*0.1
                    else:
                        ex = 2.0+np.random.rand()*0.1
                    tmpLine=tmpLine+','+str(ex)
            else:
                for j in range(totalSize):
                    if labelArray[j]==1:
                        ex = 2.0+np.random.rand()*0.1
                    else:
                        ex = 1.0+np.random.rand()*0.1
                    tmpLine=tmpLine+','+str(ex)
        elif modelName == 'model5' or modelName == 'model6' or modelName == 'model7' or modelName == 'model8':
            if i<1000:
                for j in range(totalSize):
                    if labelArray[j]==1:
                        ex = 0.0+np.random.rand()*0.1
                    elif labelArray[j]==2:
                        ex = 1.0+np.random.rand()*0.1
                    else:
                        ex = 2.0+np.random.rand()*0.1
                    tmpLine=tmpLine+','+str(ex)
            else:
                for j in range(totalSize):
                    if labelArray[j]==1:
                        ex = 2.0+np.random.rand()*0.1
                    elif labelArray[j]==2:
                        ex = 1.0+np.random.rand()*0.1
                    else:
                        ex = 0.0+np.random.rand()*0.1
                    tmpLine=tmpLine+','+str(ex)
                    
        outLines=outLines+tmpLine+'\n'

    return outLines


def selectModelLabel(modelName):
    '''
    Generate True Labels
    '''
    labelArray = []
    for j in range(totalSize):
        if modelName == 'model1':
            #if j<totalSize//10 or j>(totalSize*9//10-1):
            if j<totalSize//10 or j>(totalSize*9//10-1):
                labelArray.append(1)
            else:
                labelArray.append(2)

        # TODO
        elif modelName == 'model2':
            # TODO here
            onelabelList = (list(range(10))+list(range(11,20,1))+list(range(26,30,1))+list(range(75,80,1))+list(range(81,100,1)))
            twolabelList = ([10]+list(range(20,26,1))+list(range(30,75,1))+[81])
            if j in onelabelList:
                labelArray.append(1)
            else:
                labelArray.append(2)

        elif modelName == 'model3':
            onelabelList = (list(range(22))+list(range(23,25,1))+list(range(26,28,1))+[29]+list(range(71,73,1))+list(range(74,76,1))+list(range(77,79,1))+list(range(80,100,1)))
            twolabelList = ([22]+[25]+[28]+list(range(30,71,1))+[73]+[76]+[79])
            if j in onelabelList:
                labelArray.append(1)
            else:
                labelArray.append(2)

        elif modelName == 'model4':
            onelabelList = (list(range(33,37,1))+list(range(44,47,1))+list(range(53,57,1))+list(range(64,67,1)))
            if j<totalSize//10 or j>(totalSize*9//10-1) or (j in onelabelList):
                labelArray.append(1)
            else:
                labelArray.append(2)
        
        elif modelName == 'model5':
            if j<totalSize//10:
                labelArray.append(1)
            elif j>(totalSize*9//10-1):
                labelArray.append(3)
            else:
                labelArray.append(2)
                
        elif modelName == 'model6':
            onelabelList = (list(range(10))+list(range(11,20,1))+list(range(26,30,1)))
            twolabelList = ([10]+list(range(20,26,1))+list(range(30,75,1))+[81])
            threelabelList = (list(range(75,80,1))+list(range(81,100,1)))
            if j in onelabelList:
                labelArray.append(1)
            elif j in twolabelList:
                labelArray.append(2)
            else:
                labelArray.append(3)
                
        elif modelName == 'model7':
            onelabelList = (list(range(22))+list(range(23,25,1))+list(range(26,28,1))+[29])
            twolabelList = ([22]+[25]+[28]+list(range(30,71,1))+[73]+[76]+[79])
            threelabelList = (list(range(71,73,1))+list(range(74,76,1))+list(range(77,79,1))+list(range(80,100,1)))
            if j in onelabelList:
                labelArray.append(1)
            elif j in twolabelList:
                labelArray.append(2)
            else:
                labelArray.append(3)
                
        elif modelName == 'model8':
            onelabelList = (list(range(33,37,1))+list(range(44,47,1))+list(range(53,57,1))+list(range(64,67,1)))
            if j<totalSize//10 or (j in onelabelList):
                labelArray.append(1)
            elif j>(totalSize*9//10-1):
                labelArray.append(3)
            else:
                labelArray.append(2)

    return labelArray

modelName = 'model8'
replic = 1
expName = modelName+'_'+str(replic)
if not os.path.exists(expName):
    os.makedirs(expName)

locSize = 10 #60
useSize = locSize*2
totalSize = locSize*locSize
locList = []
for i in range(locSize):
    if i%2==0:
        for j in range(locSize):
            locList.append((i,j*2))
    else:
        for j in range(locSize):
            locList.append((i,j*2+1))

#plt.plot(locList)
arr = np.array(locList)
# plt.plot(arr)
np.save(expName+'/'+'coords_array.npy',np.array(arr))

labelArray = selectModelLabel(modelName)
outLines = selectModel(modelName,labelArray)

with open (expName+'/'+'Use_expression.csv','w') as fw:
    fw.writelines(outLines)
    fw.close()

outLabelLines = 'barcode,layer\n'
for i in range(len(labelArray)):
    outLabelLines = outLabelLines + str(i) + ',' + str(labelArray[i]) + '\n'

with open (expName+'/'+'label.csv','w') as fw:
    fw.writelines(outLabelLines)
    fw.close()
    
    
# color of plot is not corect
#     
# x_mat = []
# y_mat = []

# for i in range(arr.shape[0]):
#     x_mat.append(arr[i,0])
#     y_mat.append(arr[i,1])

# plt.scatter(x_mat,y_mat)
# plt.show()

# labelArraynp = np.array(labelArray)
# onelabelList = []
# twolabelList = []
# threelabelList = []
# for i in range(len(labelArraynp)):
#     if labelArraynp[i] == 1:
#         onelabelList.append(locList[i])
#     elif labelArraynp[i] == 2:
#         twolabelList.append(locList[i])
#     else:
#         threelabelList.append(locList[i])

# onelabelListnp = np.array(onelabelList)
# twolabelListnp = np.array(twolabelList)
# threelabelListnp = np.array(threelabelList)

# if len(threelabelListnp)!=0:
#     plt.figure(figsize=(8, 8))
#     plt.scatter(onelabelListnp[:, 0], onelabelListnp[:, 1], marker='o', cmap='coolwarm')
#     plt.scatter(twolabelListnp[:, 0], twolabelListnp[:, 1], marker='o', cmap='summer')
#     plt.scatter(threelabelListnp[:, 0], threelabelListnp[:, 1], marker='o', cmap='Set1')
# if len(threelabelListnp)==0:
#     plt.figure(figsize=(8, 8))
#     plt.scatter(onelabelListnp[:, 0], onelabelListnp[:, 1], marker='o', cmap='coolwarm')
#     plt.scatter(twolabelListnp[:, 0], twolabelListnp[:, 1], marker='o', cmap='summer')