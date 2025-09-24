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

    for feature in continuous_features:
        data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()

    """Adjust the class labels
    Using np.where()
    condition, if true value, if false value"""
    negative_label = 0
    data['binary_label'] = np.where(data['num'] == negative_label, 0 , 1)


    return data

"""Function will handle all nan values
    Numerical Continuous Values replaced with mean in relation to their label (0 or 1)
    Categorical Values will be replaced with the mode again in relation to their label
    The values of : 'ca', 'slope', and 'thal' will be replaced with unknown for now because
    i believe the nan values of those features have some meaning such as not tested or unknown"""

def handleNanValues(data):

    #print("Nan Values: ", data.isna().sum())

    """Nan Values:  
        id                0
        age               0
        sex               0
        dataset           0
        cp                0
        trestbps         59
        chol             30
        fbs              90
        restecg           2
        thalch           55
        exang            55
        oldpeak          62
        slope           309
        ca              611
        thal            486
        num               0
        binary_label      0
                            """
    
    continuous_features = ['trestbps', 'chol', 'thalch', 'oldpeak']
    for feature in continuous_features:
        for label in data['binary_label'].unique():
            mean_value = data[data['binary_label'] == label][feature].mean()
            data.loc[(data['binary_label'] == label) & (data[feature].isna()), feature] = mean_value

    categorical_features = ['fbs', 'restecg', 'exang']
    for feature in categorical_features:
        for label in data['binary_label'].unique():
            mode_value = data[data['binary_label'] == label][feature].mode()
            if not mode_value.empty: 
                mode_value = mode_value[0]
                data.loc[(data['binary_label'] == label) & (data[feature].isna()), feature] = mode_value

    for feature in ['slope', 'ca', 'thal']:
        data[feature] = data[feature].fillna("unknown")


    return data



"""Function will split into an 80% , 20% split"""
    
def train_test_splits(data):

    trainingSize = int(.8 * len(data))

    trainingData = data.iloc[:trainingSize, :]
    testingData = data.iloc[trainingSize:, :]

    """Veryify the split
    print("Training Data: ", trainingData)
    print("Testing Data: ", testingData)"""


    return trainingData,testingData



def calcCatProb(trainingData):
    labelZero = trainingData[trainingData['binary_label'] == 0]
    labelOne = trainingData[trainingData['binary_label'] == 1]

    categoricalValues = {}
    for feature in categorical_features:
        categoricalValues[feature] = trainingData[feature].unique()

    catProbs = {}

    for feature, values in categoricalValues.items():
        catProbs[feature] = {}
        k = len(values) 
        for value in values:
            # P(value | label=0)
            count_zero = len(labelZero[labelZero[feature] == value])
            pvalue_given_zero = (count_zero + 1) / (len(labelZero) + k)

            # P(value | label=1)
            count_one = len(labelOne[labelOne[feature] == value])
            pvalue_given_one = (count_one + 1) / (len(labelOne) + k)

            catProbs[feature][value] = {
                0: pvalue_given_zero,
                1: pvalue_given_one
            }

    return catProbs

def calcGaussParameters(trainingData):
    labelZero = trainingData[trainingData['binary_label'] == 0]
    labelOne = trainingData[trainingData['binary_label'] == 1]

    gaussParams = {}

    for feature in continuous_features:
        gaussParams[feature] = {

        0 : {
            "mean" : labelZero[feature].mean(),
            "std" : labelZero[feature].std(ddof = 0)
        },
        1 : {
            "mean" : labelOne[feature].mean(),
            "std" : labelOne[feature].std(ddof = 0)
        }
        }

    return gaussParams

def useGaussFormula(x, mean, std):
    exponent = np.exp(- ((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
       
def calcGauss(trainingData, gaussParams):
    gaussProbs = {}

    for feature, labelDict in gaussParams.items():
        gaussProbs[feature] = {}
        for label, stats in labelDict.items():
            mean = stats['mean']
            std = stats['std']

            values = trainingData[feature].values
            gaussProbs[feature][label] = [
                useGaussFormula(x, mean, std) for x in values
            ]

    return gaussProbs

def calcPriors(trainingData):
    priors = {
        0: len(trainingData[trainingData['binary_label'] == 0]) / len(trainingData),
        1: len(trainingData[trainingData['binary_label'] == 1]) / len(trainingData)
    }
    return priors


def predictNaiveBayes(row, priors, catProbs, gaussParams):
    posteriors = {}
    for label in [0, 1]:
        prob = priors[label]
        for feature in categorical_features:
            value = row[feature]
            prob *= catProbs[feature].get(value, {0: 1e-6, 1: 1e-6})[label]  # avoid zero prob
        for feature in continuous_features:
            mean = gaussParams[feature][label]['mean']
            std = gaussParams[feature][label]['std']
            prob *= useGaussFormula(row[feature], mean, std)
        posteriors[label] = prob
    return max(posteriors, key=posteriors.get)


def printConfusionMatrix(tp, fp, tn, fn):
    print("\n%15sActual" % "")
    print("%6s %7s %7s" % ("", "1", "0"))
    print("P%6s +--------+--------+" % "")
    print("r%6s | %-6s | %-6s |" % ("1", 'TP=' + str(tp), 'FP=' + str(fp)))
    print("e%6s +--------+--------+" % "")
    print("d%6s | %-6s | %-6s |" % ("0", 'FN=' + str(fn), 'TN=' + str(tn)))
    print(".%6s +--------+--------+\n" % "")


def getConfusionMatrix(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fn = sum((y_true == 1) & (y_pred == 0))
    return tp, fp, tn, fn


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


def showWhatHappened(testingData, predictions):
    y_true = testingData['binary_label'].values
    y_pred = np.array(predictions) 
    tp, fp, tn, fn = getConfusionMatrix(y_true, y_pred)
    printConfusionMatrix(tp, fp, tn, fn)
    print('Accuracy:  %8.5f' % getAccuracy(tp, fp, tn, fn))
    print('Precision: %8.5f' % getPrecision(tp, fp, tn, fn))
    print('Recall:    %8.5f' % getRecall(tp, fp, tn, fn))
    print('F-Measure: %8.5f' % getFScore(tp, fp, tn, fn))


if __name__ == '__main__':
    data = readdata()
    data = handleNanValues(data)
    trainingData, testingData = train_test_splits(data)

    # Calculate probabilities
    catProbs = calcCatProb(trainingData)
    gaussParameters = calcGaussParameters(trainingData)
    priors = calcPriors(trainingData)

    # Make predictions on the test set
    predictions = [
        predictNaiveBayes(row, priors, catProbs, gaussParameters)
        for _, row in testingData.iterrows()
    ]

    print("----------------------------Predictions------------------------------------")
    print("First 20 preds:", predictions[:20])
    print("Unique preds:", set(predictions))

    # Show confusion matrix and metrics
    showWhatHappened(testingData, predictions)