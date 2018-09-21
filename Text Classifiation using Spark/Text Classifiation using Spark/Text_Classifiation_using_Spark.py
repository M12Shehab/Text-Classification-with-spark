
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import (HashingTF, VectorAssembler,CountVectorizer, IDF,Tokenizer, StopWordsRemover,StringIndexer)
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.functions import length
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import sys
import gc
import os
import shutil

spark = SparkSession.builder.\
    config("spark.executor.memory", "4g")\
                  .master('local[*]') \
                  .config("spark.driver.memory","18g")\
                  .config("spark.executor.cores","4")\
                  .config("spark.python.worker.memory","4g")\
                  .config("spark.driver.maxResultSize","0")\
                  .config("spark.default.parallelism","2")\
    .appName('ML_project').getOrCreate()

#This function is used to load big json file
def read_json(filePath):
    df = spark.read.json(filePath)
    return df

def read_csv(filePath):
    df = spark.read.csv(filePath, inferSchema= True, header= True)
    df_shcema = df.withColumn("Sentiment",df.Sentiment.cast("int"))
    
    
    #final_df.printSchema()
    return df_shcema

def MachineLearning(df):
    file_dataSVM = "G:/Projects/Spark-Machine-Learning/Spark Machine Learning/Spark Machine Learning/svm/"
    data = df.select(['Summary','Sentiment']).withColumnRenamed('Sentiment','label')
    data = data.withColumn('length',length(data['Summary']))
    # Basic sentence tokenizer
    tokenizer = Tokenizer(inputCol="Summary", outputCol="words")
    wordsData = tokenizer.transform(data)
    #remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_features")
    filterd_data = remover.transform(wordsData)
    filterd_data = filterd_data.select(['label','filtered_features','length'])
    #transoform dataset to vectors
    cv = CountVectorizer(inputCol="filtered_features", outputCol="features1", minDF=2.0)
    cv_model = cv.fit(filterd_data)
    data_new = cv_model.transform(filterd_data)
    #calculate IDF for all dataset
    idf = IDF(inputCol= 'features1', outputCol = 'tf_idf')
    idfModel = idf.fit(data_new)
    rescaledData = idfModel.transform(data_new)
    #prepare data for ML spark library
    cleanUp = VectorAssembler(inputCols =['tf_idf','length'],outputCol='features')
    output = cleanUp.transform(rescaledData)
    output = output.select(['label','features'])
    #split dataset to training and testing 
    train_data, test_data = output.randomSplit([0.7,0.3],seed = 300)
    #we chose naive bayes
    nb = NaiveBayes()
    #fit data to the model
    model = nb.fit(train_data)
    #test data send to the final model
    test_results = model.transform(test_data)
    #show random 10 rows from results
    test_results.show(10)
    #evaluate the model 
    acc_eva = MulticlassClassificationEvaluator()
    acc = acc_eva.evaluate(test_results)
    print('Accuracy = {}'.format(acc))
    


   

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