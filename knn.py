import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Category features
continuous_features  = ['age', 'chol', 'oldpeak', 'thalch', 'trestbps']
categorical_features = ['ca', 'cp', 'restecg', 'slope', 'thal', 'sex', 'fbs', 'exang']

#read training data, normalize continous features using (X - mean) / std
def readdata():
    data = pd.read_csv('heart_disease_uci.csv')
    #print("Data Columns:" , data.columns)

    negative_label = 0
    data['binary_label'] = np.where(data['num'] == negative_label, 0 , 1)

    return data

# Handle NaN values and normalize continuous features using training set statistics
def handleNanValues(train, test, validation):
    continuous_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_features = ['fbs', 'restecg', 'exang', 'ca', 'slope', 'thal']
    
    # Impute continuous features with mean by label (using training set)
    for feature in continuous_features:
        for label in train['binary_label'].unique():
            mean_value = train[train['binary_label'] == label][feature].mean()
            train.loc[(train['binary_label'] == label) & (train[feature].isna()), feature] = mean_value
            test.loc[(test['binary_label'] == label) & (test[feature].isna()), feature] = mean_value
            validation.loc[(validation['binary_label'] == label) & (validation[feature].isna()), feature] = mean_value
    
    # Impute categorical features with mode by label (using training set)
    for feature in categorical_features:
        for label in train['binary_label'].unique():
            mode_value = train[train['binary_label'] == label][feature].mode()
            if not mode_value.empty:
                mode_value = mode_value[0]
                train.loc[(train['binary_label'] == label) & (train[feature].isna()), feature] = mode_value
                test.loc[(test['binary_label'] == label) & (test[feature].isna()), feature] = mode_value
                validation.loc[(validation['binary_label'] == label) & (validation[feature].isna()), feature] = mode_value
    
    # Normalize continuous features using training set statistics
    for feature in continuous_features:
        mean_value = train[feature].mean()
        std_value = train[feature].std()
        train[feature] = (train[feature] - mean_value) / std_value
        test[feature] = (test[feature] - mean_value) / std_value
        validation[feature] = (validation[feature] - mean_value) / std_value
    
    return train, test, validation

# One-hot encode categorical features using training set categories
def oneHotEncoding(train, test, validation):
    train_copy = train.copy()
    test_copy = test.copy()
    validation_copy = validation.copy()
    
    for feature in categorical_features:
        unique_vals = train[feature].unique() 
        for val in unique_vals:
            col_name = f"{feature}_{val}"
            train_copy[col_name] = (train_copy[feature] == val).astype(int)
            test_copy[col_name] = (test_copy[feature] == val).astype(int)
            validation_copy[col_name] = (validation_copy[feature] == val).astype(int)
    
    # Drop original categorical columns
    train_copy = train_copy.drop(columns=categorical_features)
    test_copy = test_copy.drop(columns=categorical_features)
    validation_copy = validation_copy.drop(columns=categorical_features)
    return train_copy, test_copy, validation_copy

# Custom stratified split to achieve 60% train, 20% test, 20% validation
def train_test_validation_splits(data):
    # Shuffle data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate by class
    class_0 = data[data['binary_label'] == 0]
    class_1 = data[data['binary_label'] == 1]
    
    # Calculate split sizes
    total_size = len(data)
    train_size = int(0.6 * total_size)
    test_size = int(0.2 * total_size)
    validation_size = total_size - train_size - test_size
    
    # Calculate class proportions
    class_0_ratio = len(class_0) / total_size
    class_1_ratio = len(class_1) / total_size
    
    # Allocate samples per class
    train_0_size = int(train_size * class_0_ratio)
    train_1_size = train_size - train_0_size
    test_0_size = int(test_size * class_0_ratio)
    test_1_size = test_size - test_0_size
    validation_0_size = len(class_0) - train_0_size - test_0_size
    validation_1_size = len(class_1) - train_1_size - test_1_size
    
    # Split each class
    train_0 = class_0.iloc[:train_0_size]
    train_1 = class_1.iloc[:train_1_size]
    test_0 = class_0.iloc[train_0_size:train_0_size + test_0_size]
    test_1 = class_1.iloc[train_1_size:train_1_size + test_1_size]
    validation_0 = class_0.iloc[train_0_size + test_0_size:]
    validation_1 = class_1.iloc[train_1_size + test_1_size:]
    
    # Combine splits
    train = pd.concat([train_0, train_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    test = pd.concat([test_0, test_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    validation = pd.concat([validation_0, validation_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return train, test, validation

def useEuclideanCalc(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


"""
Calculates Euclidean distance between a row in test and all rows in train.
This includes:
- Continuous numeric features (raw differences squared).
- One-hot encoded categorical features (0/1 mismatch treated as distance).
"""

def calcEuclideanDistance(train, test):
    # Drop non-feature columns
    train_data = train.drop(columns=['id', 'dataset', 'binary_label', 'num'])
    test_data = test.drop(columns=['id', 'dataset', 'binary_label', 'num'])
    train_targets = train['binary_label'].values  
    
    distances = []  # list to store distances row by row

    for _, row in test_data.iterrows():  # each test sample
        row_distances = []
        for _, row_train in train_data.iterrows():  # each train sample
            x = row.values
            y = row_train.values
            dist = useEuclideanCalc(x, y)
            row_distances.append(dist)
        distances.append(row_distances)
   
    """Shape = (numTest,numTrain)"""

    distances = np.array(distances)  

    
    
    return distances,train_targets

def getKNearestNeighbors(distances,targets, k = 3):

    nearestNeighbors = np.argsort(distances, axis = 1)[:,:k]

    neighborTargets = targets[nearestNeighbors]

    return neighborTargets

def knnClassify(neighborTargets):
    predictions = []

    for row in neighborTargets:
        counts = np.bincount(row)
        majorityCount = np.argmax(counts)
        predictions.append(majorityCount)

    predictions = np.array(predictions)

    return predictions

def getConfusionMatrix(test, predictions):

    yTrue = test['binary_label'].values
    yPred = np.array(predictions)

    truePositive = sum((yTrue == 1) & (yPred == 1))
    falsePositive = sum((yTrue == 0) & (yPred == 1))
    trueNegative = sum((yTrue == 0) & (yPred == 0))
    falseNegative = sum((yTrue == 1) & (yPred == 0))

    return truePositive,falsePositive,trueNegative,falseNegative

def printConfusionMatrix(tp, fp, tn, fn):
    print("\n%15sActual" % "")
    print("%6s %7s %7s" % ("", "1", "0"))
    print("P%6s +--------+--------+" % "")
    print("r%6s | %-6s | %-6s |" % ("1", 'TP='+str(tp), 'FP='+str(fp)))
    print("e%6s +--------+--------+" % "")
    print("d%6s | %-6s | %-6s |" % ("0", 'FN='+str(fn), 'TN='+str(tn)))
    print(".%6s +--------+--------+\n" % "")



def getAccuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def getPrecision(tp, fp, tn, fn):
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def getRecall(tp, fp, tn, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def getFScore(tp, fp, tn, fn):
    precision = getPrecision(tp, fp, tn, fn)
    recall = getRecall(tp, fp, tn, fn)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def showStats(tp, fp, tn, fn):
    print("Accuracy: ", round(getAccuracy(tp,fp,tn,fn),3))
    print('Precision', round(getPrecision(tp,fp,tn,fn),3))
    print('Recall: ', round(getRecall(tp,fp,tn,fn),3))
    print('F score: ', round(getFScore(tp,fp,tn,fn),3))

    return None

def testKVals(distances, targets,test):


    kVals = [1,3,5,7,9,13,20]
    for k in kVals:
        neighborTargets = getKNearestNeighbors(distances, targets, k)
        predictions = knnClassify(neighborTargets)
        tp, fp, tn, fn = getConfusionMatrix(test,predictions)
        print(f'--------------Confusion Matrix for K = {k}--------------')
        print('----------------------------------------------------------------------------')
        printConfusionMatrix(tp, fp, tn, fn)
        print(f'--------------Stats for K = {k}--------------')
        print('----------------------------------------------------------------------------')
        showStats(tp, fp, tn, fn)

    
if __name__ == '__main__':
    data = readdata()
    train,test,validation = train_test_validation_splits(data)
    train,test,validation = handleNanValues(train, test, validation)
    train,test,validation = oneHotEncoding(train,test,validation)
    distances,targets = calcEuclideanDistance(train,test)
    neighborTargets = getKNearestNeighbors(distances,targets,k=3)
    predictions = knnClassify(neighborTargets)
    tp,fp,tn,fn = getConfusionMatrix(test, predictions)
    printConfusionMatrix(tp, fp, tn, fn)
    showStats(tp, fp, tn, fn)
    testKVals(distances, targets,test)