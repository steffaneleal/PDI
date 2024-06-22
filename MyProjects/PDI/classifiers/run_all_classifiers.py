import matplotlib.pyplot as plt
import os, time
from datetime import datetime
import mlp_classifier, rf_classifier, svm_classifier

def main():
    mainStartTime = time.time()
    results = []
    modelNames = ['MLP','SVM','RF']
    print(f'[INFO] *********MLP**********.')
    result, featureName = mlp_classifier.main()
    results.append(result)
    print(f'[INFO] *********SVM**********.')
    result, featureName = svm_classifier.main()
    results.append(result)
    print(f'[INFO] *********RF**********.')
    result, featureName = rf_classifier.main()
    results.append(result)
    elapsedTime = round(time.time() - mainStartTime,2)
    print(f'[INFO] Total code execution time: {elapsedTime}s')
    plotResults(modelNames,results,featureName)

def plotResults(modelNames,results,featureName):
    fig, ax = plt.subplots()
    bar_container = ax.bar(modelNames, results,color=['red', 'green', 'blue', 'cyan'])
    ax.set_ylabel('Accuracy',weight='bold')
    ax.set_xlabel('Models',weight='bold')
    ax.set_title('Model comparison: '+featureName,fontsize=18,weight='bold')
    ax.bar_label(bar_container, fmt='{:,.2f}%')
    plt.savefig('./results/'+featureName+'-'+getCurrentFileNameAndDateTime(), dpi=300) 
    print(f'[INFO] Plotting final results done in ./results/{getCurrentFileNameAndDateTime()}')
    print(f'[INFO] Close the figure window to end the program.')
    plt.show(block=False)

def getCurrentFileNameAndDateTime():
    fileName =  os.path.basename(__file__).split('.')[0] 
    dateTime = datetime.now().strftime('-%d%m%Y-%H%M')
    return fileName+dateTime    

if __name__ == "__main__":
    main()
