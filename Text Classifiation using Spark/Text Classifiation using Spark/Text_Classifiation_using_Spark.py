
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier,LogisticRegression
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import (ChiSqSelector,HashingTF, VectorAssembler,CountVectorizer, IDF,Tokenizer, StopWordsRemover,StringIndexer, StandardScaler)
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.functions import length
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from sklearn.metrics import roc_curve, auc,confusion_matrix, classification_report
from matplotlib import pyplot as plt
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import sys
import gc
import numpy as np
import pandas as pd

spark = SparkSession.builder.\
    config("spark.executor.memory", "4g")\
                  .master('local[*]') \
                  .config("spark.driver.memory","18g")\
                  .config("spark.executor.cores","4")\
                  .config("spark.python.worker.memory","4g")\
                  .config("spark.driver.maxResultSize","0")\
                  .config("spark.default.parallelism","2")\
    .appName('ML_project').getOrCreate()
sc = spark.sparkContext
#This function is used to load big json file
def read_json(filePath):
    df = spark.read.json(filePath)
    return df

def read_csv(filePath):
    df = spark.read.csv(filePath, inferSchema= True, header= True)
    df_shcema = df.withColumn("Sentiment",df.Sentiment.cast("int"))
    
    
    #final_df.printSchema()
    return df_shcema

def DrawROC(results):
    ## prepare score-label set
    results_collect = results.collect()
    results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
    y_test = [i[1] for i in results_list]
    y_score = [i[0] for i in results_list]

    total_data = len(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_score).ravel()
    false_positives = fp / total_data
    true_positives  = tp / total_data
    false_negatives = fn / total_data
    true_negatives  = tn / total_data

    print("\t\tTrue Positive\tFalse Positive")
    print("\t\t{}\t{}".format(true_positives,false_positives))
    print("\t\tFalse Negatives\tTrue Negatives")
    print("\t\t{}\t{}".format(false_negatives,true_negatives))
    print(classification_report(y_test, y_score))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
 
    y_test = [i[1] for i in results_list]
    y_score = [i[0] for i in results_list]
 
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
 
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve  ')
    plt.legend(loc="lower right")
    plt.show()

def MachineLearning(df):
    file_dataSVM = "G:/Projects/Spark-Machine-Learning/Spark Machine Learning/Spark Machine Learning/svm/"
    data = df.select(['Summary','Sentiment']).withColumnRenamed('Sentiment','label')
    data = data.withColumn('length',length(data['Summary']))
    # Basic sentence tokenizer
    tokenizer = Tokenizer(inputCol="Summary", outputCol="words")
   
    #remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_features")
   
    #transoform dataset to vectors
    cv = HashingTF(inputCol="filtered_features", outputCol="features1", numFeatures=1000)
    
    #calculate IDF for all dataset
    idf = IDF(inputCol= 'features1', outputCol = 'tf_idf')
    
    normalizer = StandardScaler(inputCol="tf_idf", outputCol="normFeatures", withStd=True, withMean=False)
    selector = ChiSqSelector(numTopFeatures=150, featuresCol="normFeatures",
                         outputCol="selectedFeatures", labelCol="label")
    #prepare data for ML spark library
    cleanUp = VectorAssembler(inputCols =['selectedFeatures'],outputCol='features')
    # Normalize each Vector using $L^1$ norm.
    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf,normalizer,selector,cleanUp])
    pipelineModel = pipeline.fit(data)
    data = pipelineModel.transform(data)
    data.printSchema()
    train_data, test_data = data.randomSplit([0.7,0.3],seed=2018)

    lr = LogisticRegression(featuresCol="features", labelCol='label')
    lrModel = lr.fit(train_data)
    beta = np.sort(lrModel.coefficients)
    plt.plot(beta)
    plt.ylabel('Beta Coefficients')
    plt.show()

    trainingSummary = lrModel.summary
    roc = trainingSummary.roc.toPandas()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))



    pr = trainingSummary.pr.toPandas()
    plt.plot(pr['recall'],pr['precision'])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    predictions = lrModel.transform(test_data)
    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))
    #we chose naive bayes family="binomial"
    
    

def MachineLearning2(df):
    file_dataSVM = "G:/Projects/Spark-Machine-Learning/Spark Machine Learning/Spark Machine Learning/svm/"
    data = df.select(['Summary','Sentiment']).withColumnRenamed('Sentiment','label')
    data = data.withColumn('length',length(data['Summary']))
    # Basic sentence tokenizer
    tokenizer = Tokenizer(inputCol="Summary", outputCol="words")
   
    #remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_features")
   
    #transoform dataset to vectors
    cv = CountVectorizer(inputCol="filtered_features", outputCol="features1",vocabSize=1000, minDF=1000.0)
  
    #calculate IDF for all dataset
    idf = IDF(inputCol= 'features1', outputCol = 'tf_idf')
   
    #prepare data for ML spark library
    cleanUp = VectorAssembler(inputCols =['tf_idf','length'],outputCol='features')
   
    train_data, test_data = data.randomSplit([0.7,0.3],1)


    #we chose naive bayes
    nb = NaiveBayes(smoothing=2.0,featuresCol="features",labelCol='label',predictionCol ="prediction")

    paramGrid = ParamGridBuilder().build()
    numFolds = 10
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",labelCol="label") # + other params as in Scala    

    #add pipline technique
    pipeline = Pipeline(stages=[tokenizer, remover, cv,idf,cleanUp,nb])
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=numFolds)

   
    # Fit the pipeline to training documents.
    model = crossval.fit(train_data)
    #fit data to the model
    #model = nb.fit(train_data)
    model_path = 'C:/Users/Shehab/Source/Repos/Text-Classification-with-spark/Text Classifiation using Spark/Text Classifiation using Spark/model/'
    try:
        model.save(model_path )
    except:
        print('Error folder of model already exits')

    #test data send to the final model
    test_results = model.transform(test_data)
    #test_results.show(10)
    results = test_results.select(['prediction', 'label'])
    df_shcema = results.withColumn("prediction",results.prediction.cast("string"))
    #draw ROC curve
    DrawROC(df_shcema)
    #show random 10 rows from results
    test_results.show(10)
    #evaluate the model 
    #acc_eva = MulticlassClassificationEvaluator()
    #acc = acc_eva.evaluate(test_results)
    #print('Accuracy = {}'.format(acc))
   

def free_memory():
    c = gc.collect()
    print('Free {} memory allocation..'.format(c))

def main():
    #here set your data location path
    dataPath = 'Musical_Instruments_5.json'# this dataset is from http://jmcauley.ucsd.edu/data/amazon/links.html
    dataCsv = 'movie_reviews.csv'#
   
    #read dataset from csv
    df = read_csv(dataCsv)
    df.printSchema()
    MachineLearning(df)

    spark.stop()
if __name__ =='__main__':
    main() 