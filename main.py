from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, DoubleType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import joblib

# Create a SparkSession
spark = SparkSession.builder.appName("LoadGzJson").getOrCreate()

# Define the schema of the dataset
schema = StructType([
    StructField("asin", StringType(), True),
    StructField("overall", DoubleType(), True),
    StructField("vote", StringType(), True),
    StructField("verified", StringType(), True),
    StructField("reviewTime", StringType(), True),
    StructField("reviewerID", StringType(), True),
    StructField("reviewerName", StringType(), True),
    StructField("reviewText", StringType(), True),
    StructField("summary", StringType(), True)
])

# Read the JSON file with the defined schema
json_file = spark.read.schema(schema).json("/media/abdul/2CE0E9CBE0E99AFA/Users/Abdul/Desktop")

# Select the first 1000 rows
first_1000_rows = json_file.limit(10000)

# Remove duplicates and null values
first_1000_rows_clean = first_1000_rows.dropDuplicates(["reviewerID", "reviewText"]).na.drop(subset=["asin", "reviewerID", "reviewText"])

# Convert the reviewerID and asin to a numeric index
reviewer_indexer = StringIndexer(inputCol="reviewerID", outputCol="reviewerIndex")
asin_indexer = StringIndexer(inputCol="asin", outputCol="asinIndex")

# Prepare the dataset for collaborative filtering
ratings = first_1000_rows_clean.select(col("reviewerID"), col("asin"), col("overall").cast("float"))
pipeline = Pipeline(stages=[reviewer_indexer, asin_indexer])
indexed_ratings = pipeline.fit(ratings).transform(ratings).select(col("reviewerIndex").cast("int"), col("asinIndex").cast("int"), "overall")

# Split the dataset into training and testing sets
(train, test) = indexed_ratings.randomSplit([0.8, 0.2], seed=42)

# Train the ALS model
als = ALS(userCol="reviewerIndex", itemCol="asinIndex", ratingCol="overall", nonnegative=True)
model = als.fit(train)

# Save the model using joblib
import mlflow.spark

# Save the model using mlflow
mlflow.spark.save_model(model, "/home/abdul/PycharmProjects/pythonProject5")

# Load the model using mlflow
loaded_model = mlflow.spark.load_model("/home/abdul/PycharmProjects/pythonProject5")

# Evaluate the model using the testing set
evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall", predictionCol="prediction")
predictions = loaded_model.transform(test)
predictions = predictions.na.drop()
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
predictions.select("overall", "prediction").show()

# Make recommendations for all users
recommendations = model.recommendForAllUsers(10)

# Input user ID
user_id = input("Enter user ID: ")

# Make sure to convert user_id to int as input returns a string
user_id = int(user_id)

# Get recommendations for the user
user_recommendations = recommendations.filter(recommendations.reviewerIndex == user_id)

# Now we need to convert the recommended asinIndex back to original
# Now we need to convert the recommended asinIndex back to original ASIN
asin_indexer_model = asin_indexer.fit(ratings)
indexed_ratings = asin_indexer_model.transform(ratings)

asin_index_to_string = asin_indexer_model.labels  # This is a list of ASINs

# Convert indices back to ASINs
asin_recommendations = [asin_index_to_string[i] for i in user_recommendations.select("recommendations.asinIndex").first()["asinIndex"]]
print("Recommended ASINs for the user: ", asin_recommendations)
