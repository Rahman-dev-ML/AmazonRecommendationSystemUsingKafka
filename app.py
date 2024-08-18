import mlflow.spark
from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, DoubleType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pymongo import MongoClient

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
first_1000_rows_clean = first_1000_rows.dropDuplicates(["reviewerID", "reviewText"]).na.drop(
    subset=["asin", "reviewerID", "reviewText"])

# Convert the reviewerID and asin to a numeric index
reviewer_indexer = StringIndexer(inputCol="reviewerID", outputCol="reviewerIndex")
asin_indexer = StringIndexer(inputCol="asin", outputCol="asinIndex")

# Prepare the dataset for collaborative filtering
ratings = first_1000_rows_clean.select(col("reviewerID"), col("asin"), col("overall").cast("float"))
asin_indexer_model = asin_indexer.fit(ratings)
indexed_ratings = asin_indexer_model.transform(ratings)

# Load the ALS model using mlflow
loaded_model = mlflow.spark.load_model("/home/abdul/PycharmProjects/pythonProject5")

# Extract the ALS model from the loaded model
als_model = loaded_model.stages[-1]

# Create a Flask application
app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["recommendationDB"]
collection = db["recommendations"]


# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define the route to handle the form submission
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_id = int(request.form['user_id'])

    # Get recommendations for the user
    user_recommendations = als_model.recommendForUserSubset(spark.createDataFrame([(user_id,)], ["reviewerIndex"]), 10)

    # Convert indices back to ASINs
    asin_recommendations = [asin_indexer_model.labels[i.asinIndex] for i in
                            user_recommendations.first()["recommendations"]]

    # Store the user ID and the recommendations in MongoDB
    recommendation_document = {"user_id": user_id, "recommendations": asin_recommendations}
    collection.insert_one(recommendation_document)

    return render_template('recommendations.html', asin_recommendations=asin_recommendations)


# Run the Flask application
if __name__ == '__main__':
    app.run()
