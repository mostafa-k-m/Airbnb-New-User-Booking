##  Airbnb new user booking prediction using pyspatk and spark-ml
link for data: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings


```python
import os
os.environ["JAVA_HOME"] = "/usr/lib64/jvm/jre-1.8.0-openjdk"
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Project3 - Airbnb") \
    .config("spark.driver.memory", "15g") \
    .getOrCreate()

```


```python
from pyspark.sql.types import LongType, StringType, StructField, StructType, BooleanType, ArrayType, IntegerType, DateType, FloatType
```

## Loading the train dataset


```python
schema_Train_Users = StructType([
    StructField("id", StringType(), False),
    StructField("date_account_created", DateType(), True),
    StructField("timestamp_first_active", StringType(), True),
    StructField("date_first_booking", DateType(), True),
    StructField("gender", StringType(), True),
    StructField("age", FloatType(), True),
    StructField("signup_method", StringType(), True),
    StructField("signup_flow", FloatType(), True),
    StructField("language", StringType(), True),
    StructField("affiliate_channel", StringType(), True),
    StructField("affiliate_provider", StringType(), True),
    StructField("first_affiliate_tracked", StringType(), True),
    StructField("signup_app", StringType(), True),
    StructField("first_device_type", StringType(), True),
    StructField("first_browser", StringType(), True),
    StructField("country_destination", StringType(), True)])
```


```python
train_users_raw = spark.read\
            .format("csv")\
            .schema(schema_Train_Users)\
            .option("header", "true")\
            .load("./Dataset/train_users_2.csv")
```


```python
train_users_raw.show(5)
```

    +----------+--------------------+----------------------+------------------+---------+----+-------------+-----------+--------+-----------------+------------------+-----------------------+----------+-----------------+-------------+-------------------+
    |        id|date_account_created|timestamp_first_active|date_first_booking|   gender| age|signup_method|signup_flow|language|affiliate_channel|affiliate_provider|first_affiliate_tracked|signup_app|first_device_type|first_browser|country_destination|
    +----------+--------------------+----------------------+------------------+---------+----+-------------+-----------+--------+-----------------+------------------+-----------------------+----------+-----------------+-------------+-------------------+
    |gxn3p5htnn|          2010-06-28|        20090319043255|              null|-unknown-|null|     facebook|        0.0|      en|           direct|            direct|              untracked|       Web|      Mac Desktop|       Chrome|                NDF|
    |820tgsjxq7|          2011-05-25|        20090523174809|              null|     MALE|38.0|     facebook|        0.0|      en|              seo|            google|              untracked|       Web|      Mac Desktop|       Chrome|                NDF|
    |4ft3gnwmtx|          2010-09-28|        20090609231247|        2010-08-02|   FEMALE|56.0|        basic|        3.0|      en|           direct|            direct|              untracked|       Web|  Windows Desktop|           IE|                 US|
    |bjjt8pjhuk|          2011-12-05|        20091031060129|        2012-09-08|   FEMALE|42.0|     facebook|        0.0|      en|           direct|            direct|              untracked|       Web|      Mac Desktop|      Firefox|              other|
    |87mebub9p4|          2010-09-14|        20091208061105|        2010-02-18|-unknown-|41.0|        basic|        0.0|      en|           direct|            direct|              untracked|       Web|      Mac Desktop|       Chrome|                 US|
    +----------+--------------------+----------------------+------------------+---------+----+-------------+-----------+--------+-----------------+------------------+-----------------------+----------+-----------------+-------------+-------------------+
    only showing top 5 rows
    


## Loading the sessions dataset


```python
schema_sessions_raw = StructType([
    StructField("user_id", StringType(), False),
    StructField("action", StringType(), True),
    StructField("action_type", DateType(), True),
    StructField("action_detail", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("secs_elapsed", FloatType(), True)])
```


```python
sessions_raw = spark.read\
            .format("csv")\
            .option("header", "true")\
            .load("./Dataset/sessions.csv")
```


```python
sessions_raw.show(5)
```

    +----------+--------------+-----------+-------------------+---------------+------------+
    |   user_id|        action|action_type|      action_detail|    device_type|secs_elapsed|
    +----------+--------------+-----------+-------------------+---------------+------------+
    |d1mm9tcy42|        lookup|       null|               null|Windows Desktop|       319.0|
    |d1mm9tcy42|search_results|      click|view_search_results|Windows Desktop|     67753.0|
    |d1mm9tcy42|        lookup|       null|               null|Windows Desktop|       301.0|
    |d1mm9tcy42|search_results|      click|view_search_results|Windows Desktop|     22141.0|
    |d1mm9tcy42|        lookup|       null|               null|Windows Desktop|       435.0|
    +----------+--------------+-----------+-------------------+---------------+------------+
    only showing top 5 rows
    


## Loading the countries dataset


```python
schema_countries_raw = StructType([
    StructField("country_destination", StringType(), False),
    StructField("lat_destination", FloatType(), True),
    StructField("lng_destination", FloatType(), True),
    StructField("distance_km", FloatType(), True),
    StructField("destination_km2", FloatType(), True),
    StructField("destination_language", StringType(), True),
    StructField("language_levenshtein_distance", FloatType(), True)])
```


```python
countries_raw = spark.read\
            .format("csv")\
            .option("header", "true")\
            .load("./Dataset/countries.csv")
```


```python
countries_raw.show()
```

    +-------------------+---------------+---------------+-----------+---------------+---------------------+-----------------------------+
    |country_destination|lat_destination|lng_destination|distance_km|destination_km2|destination_language |language_levenshtein_distance|
    +-------------------+---------------+---------------+-----------+---------------+---------------------+-----------------------------+
    |                 AU|     -26.853388|      133.27516|  15297.744|      7741220.0|                  eng|                          0.0|
    |                 CA|      62.393303|     -96.818146|  2828.1333|      9984670.0|                  eng|                          0.0|
    |                 DE|      51.165707|      10.452764|   7879.568|       357022.0|                  deu|                        72.61|
    |                 ES|      39.896027|     -2.4876945|   7730.724|       505370.0|                  spa|                        92.25|
    |                 FR|      46.232193|       2.209667|   7682.945|       643801.0|                  fra|                        92.06|
    |                 GB|       54.63322|     -3.4322774|   6883.659|       243610.0|                  eng|                          0.0|
    |                 IT|       41.87399|      12.564167|   8636.631|       301340.0|                  ita|                         89.4|
    |                 NL|      52.133057|        5.29525|  7524.3203|        41543.0|                  nld|                        63.22|
    |                 PT|      39.553444|      -7.839319|  7355.2534|        92090.0|                  por|                        95.45|
    |                 US|      36.966427|      -95.84403|        0.0|      9826675.0|                  eng|                          0.0|
    +-------------------+---------------+---------------+-----------+---------------+---------------------+-----------------------------+
    


## Calculating the total session time for each user


```python
from pyspark.sql.functions import sum as _sum
session_time = sessions_raw.groupby("user_id").agg(_sum('secs_elapsed').alias('sum_secs_elapsed'))
```


```python
session_time.show()
```

    +----------+----------------+
    |   user_id|sum_secs_elapsed|
    +----------+----------------+
    |de3scomvop|          1051.0|
    |9nut71te0s|       1659715.0|
    |zlv8f1qg2g|       1155388.0|
    |srykgkylee|           246.0|
    |funlgmcmr3|         54747.0|
    |mzduh7va3m|       1483785.0|
    |zds4xn9jvb|       4837348.0|
    |s5hieu20bh|        631247.0|
    |n2utn4z7pk|       2993479.0|
    |e766mg6ku1|        259326.0|
    |xfpn2xw6b6|        500275.0|
    |thkobfxs30|        480322.0|
    |fvjgmiax3d|         96818.0|
    |2gv2kfvseu|        401193.0|
    |ott06joxd2|       1292370.0|
    |e2zoe02zd5|        642272.0|
    |f0cnhta47g|       1338892.0|
    |spv23uq1cb|       1567138.0|
    |5ounyry4bv|        162051.0|
    |sl81fx9peb|        182636.0|
    +----------+----------------+
    only showing top 20 rows
    


## Adding the total Session time to the training dataset 


```python
train_users_raw = train_users_raw.join(session_time, train_users_raw["id"] == session_time["user_id"],how='left_outer').select(train_users_raw["*"],session_time["sum_secs_elapsed"])
```


```python
train_users_raw.show(5)
```

    +----------+--------------------+----------------------+------------------+---------+----+-------------+-----------+--------+-----------------+------------------+-----------------------+----------+-----------------+-------------+-------------------+----------------+
    |        id|date_account_created|timestamp_first_active|date_first_booking|   gender| age|signup_method|signup_flow|language|affiliate_channel|affiliate_provider|first_affiliate_tracked|signup_app|first_device_type|first_browser|country_destination|sum_secs_elapsed|
    +----------+--------------------+----------------------+------------------+---------+----+-------------+-----------+--------+-----------------+------------------+-----------------------+----------+-----------------+-------------+-------------------+----------------+
    |01r3iatdvv|          2014-02-11|        20140211202128|        2014-02-12|-unknown-|null|        basic|        0.0|      en|           direct|            direct|              untracked|       Web|      Mac Desktop|       Chrome|                 US|        813485.0|
    |02sgboyndc|          2014-04-29|        20140429213851|              null|-unknown-|null|        basic|        0.0|      en|        sem-brand|            google|              untracked|       Web|      Mac Desktop|       Safari|                NDF|        758902.0|
    |03c7ihv5r8|          2014-05-07|        20140507223046|              null|     MALE|47.0|     facebook|        0.0|      en|        sem-brand|            google|                    omg|       Web|  Windows Desktop|           IE|                NDF|       3003287.0|
    |05xkkfxs5v|          2014-04-06|        20140406183244|              null|     MALE|26.0|     facebook|        0.0|      en|      remarketing|            google|              untracked|       Web|      Mac Desktop|       Chrome|                NDF|        106376.0|
    |08bys9zpkj|          2013-07-04|        20130704223816|              null|   FEMALE|50.0|     facebook|        0.0|      en|    sem-non-brand|            google|              untracked|       Web|      Mac Desktop|       Safari|                NDF|            null|
    +----------+--------------------+----------------------+------------------+---------+----+-------------+-----------+--------+-----------------+------------------+-----------------------+----------+-----------------+-------------+-------------------+----------------+
    only showing top 5 rows
    



```python
train_users_raw.count()
```




    213451



## Extracting the month of the account creation, and converting gender to lower case


```python
from pyspark.sql.functions import year, month, lower, col
train_users_raw = train_users_raw.withColumn("month_account_created", month("date_account_created"))
train_users_raw = train_users_raw.withColumn("gender", lower(col('gender')))
```

## Dropping unneeded columns from train_users dataset
#### we dropped the dates (date_account_created, timestamp_first_active) but retained the month from date_account_created to account for any seasonal changes.
#### we droped date_first_booking because it is meaningless considering that our test users never made a booking.
#### we dropped first_affiliate_tracked because the two other columns affiliate_channel and affiliate_provider provide enough information about the affiliate.
#### we dropped first_browser because it had too many categories and we already know the signup_app and first_device_type.


```python
train_columnsToDrop = ['id', 'date_account_created', 'timestamp_first_active', 'date_first_booking', 'first_affiliate_tracked', 'first_browser']

train_users_raw = train_users_raw.select([column for column in train_users_raw.columns if column not in train_columnsToDrop])
```


```python
train_users_raw.show(5)
```

    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    |   gender| age|signup_method|signup_flow|language|affiliate_channel|affiliate_provider|signup_app|first_device_type|country_destination|sum_secs_elapsed|month_account_created|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    |-unknown-|null|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|        813485.0|                    2|
    |-unknown-|null|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|        758902.0|                    4|
    |     male|47.0|     facebook|        0.0|      en|        sem-brand|            google|       Web|  Windows Desktop|                NDF|       3003287.0|                    5|
    |     male|26.0|     facebook|        0.0|      en|      remarketing|            google|       Web|      Mac Desktop|                NDF|        106376.0|                    4|
    |   female|50.0|     facebook|        0.0|      en|    sem-non-brand|            google|       Web|      Mac Desktop|                NDF|            null|                    7|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    only showing top 5 rows
    


## Finding categorical columns
#### We will exclude the last column from this list because it is the label column and we will pass this list in the one hot encoder later.


```python
cat_cols = [item[0] for item in train_users_raw.dtypes if item[1].startswith('string')] 
cat_cols.pop(-1)
cat_cols
```




    ['gender',
     'signup_method',
     'language',
     'affiliate_channel',
     'affiliate_provider',
     'signup_app',
     'first_device_type']



## Finding numerical columns


```python
num_cols = [item[0] for item in train_users_raw.dtypes if item[1].startswith('int') | item[1].startswith('double') | item[1].startswith('float')] 
num_cols
```




    ['age', 'signup_flow', 'sum_secs_elapsed', 'month_account_created']



## Finding which columns contain null or unknown values
#### No nulls found in categorical columns


```python
from pyspark.sql.functions import col

cat_null_cols = [column for column in cat_cols if train_users_raw.where(col(column).isNull()).count() > 0]
cat_null_cols
```




    []



## Now let's find numerical columns with null values


```python
from pyspark.sql.functions import lit
num_null_cols = [column for column in num_cols if train_users_raw.filter(col(column).isNull() | col(column).eqNullSafe(0)).count() > 0]
num_null_cols
```




    ['age', 'signup_flow', 'sum_secs_elapsed']



#### Replacing null ages with 0 for now.
#### Replacing null web browsing time with 0, because it means the user didn't search
#### also replacing null signup_flow values with 0


```python
train_users_raw = train_users_raw.fillna(0, subset=['age'])
train_users_raw = train_users_raw.fillna(0, subset=['sum_secs_elapsed'])
train_users_raw = train_users_raw.fillna(0, subset=['signup_flow'])
```


```python
train_users_raw.show(5)
```

    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    |   gender| age|signup_method|signup_flow|language|affiliate_channel|affiliate_provider|signup_app|first_device_type|country_destination|sum_secs_elapsed|month_account_created|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    |-unknown-| 0.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|        813485.0|                    2|
    |-unknown-| 0.0|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|        758902.0|                    4|
    |     male|47.0|     facebook|        0.0|      en|        sem-brand|            google|       Web|  Windows Desktop|                NDF|       3003287.0|                    5|
    |     male|26.0|     facebook|        0.0|      en|      remarketing|            google|       Web|      Mac Desktop|                NDF|        106376.0|                    4|
    |   female|50.0|     facebook|        0.0|      en|    sem-non-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    7|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    only showing top 5 rows
    



```python
train_users_raw.describe().show()
```

    +-------+---------+------------------+-------------+------------------+--------+-----------------+------------------+----------+-----------------+-------------------+------------------+---------------------+
    |summary|   gender|               age|signup_method|       signup_flow|language|affiliate_channel|affiliate_provider|signup_app|first_device_type|country_destination|  sum_secs_elapsed|month_account_created|
    +-------+---------+------------------+-------------+------------------+--------+-----------------+------------------+----------+-----------------+-------------------+------------------+---------------------+
    |  count|   213451|            213451|       213451|            213451|  213451|           213451|            213451|    213451|           213451|             213451|            213451|               213451|
    |   mean|     null| 29.19376812476868|         null|3.2673868944160485|    null|             null|              null|      null|             null|               null| 523648.3010058515|   6.0224594871890975|
    | stddev|     null|121.82235632889513|         null| 7.637706869435076|    null|             null|              null|      null|             null|               null|1335853.6689385953|     3.23668965044066|
    |    min|-unknown-|               0.0|        basic|               0.0|      ca|              api|             baidu|   Android|    Android Phone|                 AU|               0.0|                    1|
    |    max|    other|            2014.0|       google|              25.0|      zh|              seo|            yandex|       iOS|           iPhone|              other|       3.8221363E7|                   12|
    +-------+---------+------------------+-------------+------------------+--------+-----------------+------------------+----------+-----------------+-------------------+------------------+---------------------+
    


## Correcting age values
#### for now, we will set all ages below 14 and above 90 to 0
#### if the age is between 1940 and 2001 we assumed that it was the date of birth and calculated the age = 2015 - date of birth.
#### all other values are set to 0.


```python
from pyspark.sql.functions import udf
def age_corrector(age):
    if age > 13 and age < 90:
        return age
    elif age > 1940 and age <2001:
        return 2015.0-age
    else:
        return 0.0
correct_age = udf(age_corrector)
```


```python
train_users_raw = train_users_raw.withColumn("age", correct_age(train_users_raw['age']))
train_users_raw.show()
```

    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    |   gender| age|signup_method|signup_flow|language|affiliate_channel|affiliate_provider|signup_app|first_device_type|country_destination|sum_secs_elapsed|month_account_created|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    |-unknown-| 0.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|        813485.0|                    2|
    |-unknown-| 0.0|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|        758902.0|                    4|
    |     male|47.0|     facebook|        0.0|      en|        sem-brand|            google|       Web|  Windows Desktop|                NDF|       3003287.0|                    5|
    |     male|26.0|     facebook|        0.0|      en|      remarketing|            google|       Web|      Mac Desktop|                NDF|        106376.0|                    4|
    |   female|50.0|     facebook|        0.0|      en|    sem-non-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    7|
    |-unknown-| 0.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|              other|             0.0|                    3|
    |     male|41.0|        basic|       25.0|      en|           direct|            direct|       iOS|           iPhone|                NDF|             0.0|                   10|
    |-unknown-| 0.0|        basic|        2.0|      en|           direct|            direct|       Web|           iPhone|                NDF|             0.0|                    2|
    |   female|53.0|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    1|
    |     male|49.0|     facebook|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                NDF|         21549.0|                    3|
    |-unknown-| 0.0|        basic|        0.0|      fr|        sem-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    4|
    |-unknown-| 0.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                NDF|        518880.0|                    5|
    |-unknown-| 0.0|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|       2609554.0|                    4|
    |     male|29.0|        basic|       25.0|      en|           direct|            direct|       iOS|    Other/Unknown|              other|             0.0|                   12|
    |   female|54.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|             0.0|                    8|
    |     male|33.0|     facebook|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    4|
    |     male|26.0|     facebook|        0.0|      en|           direct|            direct|       Web|  Windows Desktop|              other|             0.0|                    5|
    |-unknown-| 0.0|       google|       12.0|      en|           direct|            direct|   Android|    Android Phone|                NDF|          8773.0|                    5|
    |-unknown-| 0.0|        basic|        3.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|             0.0|                    6|
    |-unknown-| 0.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                NDF|             0.0|                    4|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    only showing top 20 rows
    


#### We know that all the test users are from usa so we will calculate the mean age of usa citizens


```python
import pandas as pd
age_gender_bkts = pd.read_csv('./Dataset/age_gender_bkts.csv')
age_gender_bkts = age_gender_bkts[age_gender_bkts.country_destination == 'US']
sum_age_times_population_in_thousands = 0
for index,row in age_gender_bkts.iterrows():
    if '+' in row['age_bucket']:
        sum_age_times_population_in_thousands += 100*row['population_in_thousands']
    else:
        age_split = [int(i) for i in row['age_bucket'].split('-')]
        mean_age_for_bucket = (age_split[0]+age_split[1])/2
        sum_age_times_population_in_thousands += mean_age_for_bucket*row['population_in_thousands']
mean_age = sum_age_times_population_in_thousands/age_gender_bkts.population_in_thousands.sum() 
mean_age
```




    38.132346247062735



#### We will replace all 0 values with the mean_age for usa


```python
correct_zero_age = udf(lambda value: value+float(round(mean_age)) if value == 0.0 else value, FloatType())
train_users_raw = train_users_raw.withColumn("age", correct_zero_age(train_users_raw['age']))
```


```python
train_users_raw.show()
```

    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    |   gender| age|signup_method|signup_flow|language|affiliate_channel|affiliate_provider|signup_app|first_device_type|country_destination|sum_secs_elapsed|month_account_created|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    |-unknown-|38.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|        813485.0|                    2|
    |-unknown-|38.0|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|        758902.0|                    4|
    |     male|47.0|     facebook|        0.0|      en|        sem-brand|            google|       Web|  Windows Desktop|                NDF|       3003287.0|                    5|
    |     male|26.0|     facebook|        0.0|      en|      remarketing|            google|       Web|      Mac Desktop|                NDF|        106376.0|                    4|
    |   female|50.0|     facebook|        0.0|      en|    sem-non-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    7|
    |-unknown-|38.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|              other|             0.0|                    3|
    |     male|41.0|        basic|       25.0|      en|           direct|            direct|       iOS|           iPhone|                NDF|             0.0|                   10|
    |-unknown-|38.0|        basic|        2.0|      en|           direct|            direct|       Web|           iPhone|                NDF|             0.0|                    2|
    |   female|53.0|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    1|
    |     male|49.0|     facebook|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                NDF|         21549.0|                    3|
    |-unknown-|38.0|        basic|        0.0|      fr|        sem-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    4|
    |-unknown-|38.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                NDF|        518880.0|                    5|
    |-unknown-|38.0|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|       2609554.0|                    4|
    |     male|29.0|        basic|       25.0|      en|           direct|            direct|       iOS|    Other/Unknown|              other|             0.0|                   12|
    |   female|54.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|             0.0|                    8|
    |     male|33.0|     facebook|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    4|
    |     male|26.0|     facebook|        0.0|      en|           direct|            direct|       Web|  Windows Desktop|              other|             0.0|                    5|
    |-unknown-|38.0|       google|       12.0|      en|           direct|            direct|   Android|    Android Phone|                NDF|          8773.0|                    5|
    |-unknown-|38.0|        basic|        3.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|             0.0|                    6|
    |-unknown-|38.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                NDF|             0.0|                    4|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+
    only showing top 20 rows
    


## Bulding a pipeline to encode features


```python
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator
from pyspark.ml import Pipeline
indexers = [StringIndexer(inputCol=column,outputCol=column + '_indexed', handleInvalid='keep') for column in cat_cols]
encoder = OneHotEncoderEstimator(inputCols=[column+'_indexed' for column in cat_cols], outputCols=[column+'_encoded' for column in cat_cols])
pipeline = Pipeline(stages=indexers + [encoder])
pipeline = pipeline.fit(train_users_raw)
train_users_raw = pipeline.transform(train_users_raw)
train_users_raw = train_users_raw.drop(*[column+'_indexed' for column in cat_cols])
train_users_raw.fillna(0.0)
train_users_raw.show(5)
```

    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+-------------------------+--------------+---------------------+--------------------------+-------------------------+------------------+----------------+
    |   gender| age|signup_method|signup_flow|language|affiliate_channel|affiliate_provider|signup_app|first_device_type|country_destination|sum_secs_elapsed|month_account_created|affiliate_channel_encoded|gender_encoded|signup_method_encoded|affiliate_provider_encoded|first_device_type_encoded|signup_app_encoded|language_encoded|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+-------------------------+--------------+---------------------+--------------------------+-------------------------+------------------+----------------+
    |-unknown-|38.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|        813485.0|                    2|            (8,[0],[1.0])| (4,[0],[1.0])|        (3,[0],[1.0])|            (18,[0],[1.0])|            (9,[0],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|
    |-unknown-|38.0|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|        758902.0|                    4|            (8,[1],[1.0])| (4,[0],[1.0])|        (3,[0],[1.0])|            (18,[1],[1.0])|            (9,[0],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|
    |     male|47.0|     facebook|        0.0|      en|        sem-brand|            google|       Web|  Windows Desktop|                NDF|       3003287.0|                    5|            (8,[1],[1.0])| (4,[2],[1.0])|        (3,[1],[1.0])|            (18,[1],[1.0])|            (9,[1],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|
    |     male|26.0|     facebook|        0.0|      en|      remarketing|            google|       Web|      Mac Desktop|                NDF|        106376.0|                    4|            (8,[7],[1.0])| (4,[2],[1.0])|        (3,[1],[1.0])|            (18,[1],[1.0])|            (9,[0],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|
    |   female|50.0|     facebook|        0.0|      en|    sem-non-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    7|            (8,[2],[1.0])| (4,[1],[1.0])|        (3,[1],[1.0])|            (18,[1],[1.0])|            (9,[0],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+-------------------------+--------------+---------------------+--------------------------+-------------------------+------------------+----------------+
    only showing top 5 rows
    


## Assembling the feature vector for the training dataset


```python
from pyspark.ml.feature import VectorAssembler
train_users_raw = train_users_raw.withColumn("age", train_users_raw["age"].cast(FloatType()))

assembler = VectorAssembler(inputCols=[i for i in train_users_raw.columns if (i !='country_destination') and (i not in cat_cols)], outputCol='features')
train_users_raw = assembler.transform(train_users_raw)

train_users_raw.show(5)
```

    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+-------------------------+--------------+---------------------+--------------------------+-------------------------+------------------+----------------+--------------------+
    |   gender| age|signup_method|signup_flow|language|affiliate_channel|affiliate_provider|signup_app|first_device_type|country_destination|sum_secs_elapsed|month_account_created|affiliate_channel_encoded|gender_encoded|signup_method_encoded|affiliate_provider_encoded|first_device_type_encoded|signup_app_encoded|language_encoded|            features|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+-------------------------+--------------+---------------------+--------------------------+-------------------------+------------------+----------------+--------------------+
    |-unknown-|38.0|        basic|        0.0|      en|           direct|            direct|       Web|      Mac Desktop|                 US|        813485.0|                    2|            (8,[0],[1.0])| (4,[0],[1.0])|        (3,[0],[1.0])|            (18,[0],[1.0])|            (9,[0],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|(75,[0,2,3,4,12,1...|
    |-unknown-|38.0|        basic|        0.0|      en|        sem-brand|            google|       Web|      Mac Desktop|                NDF|        758902.0|                    4|            (8,[1],[1.0])| (4,[0],[1.0])|        (3,[0],[1.0])|            (18,[1],[1.0])|            (9,[0],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|(75,[0,2,3,5,12,1...|
    |     male|47.0|     facebook|        0.0|      en|        sem-brand|            google|       Web|  Windows Desktop|                NDF|       3003287.0|                    5|            (8,[1],[1.0])| (4,[2],[1.0])|        (3,[1],[1.0])|            (18,[1],[1.0])|            (9,[1],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|(75,[0,2,3,5,14,1...|
    |     male|26.0|     facebook|        0.0|      en|      remarketing|            google|       Web|      Mac Desktop|                NDF|        106376.0|                    4|            (8,[7],[1.0])| (4,[2],[1.0])|        (3,[1],[1.0])|            (18,[1],[1.0])|            (9,[0],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|(75,[0,2,3,11,14,...|
    |   female|50.0|     facebook|        0.0|      en|    sem-non-brand|            google|       Web|      Mac Desktop|                NDF|             0.0|                    7|            (8,[2],[1.0])| (4,[1],[1.0])|        (3,[1],[1.0])|            (18,[1],[1.0])|            (9,[0],[1.0])|     (4,[0],[1.0])|  (25,[0],[1.0])|(75,[0,3,6,13,17,...|
    +---------+----+-------------+-----------+--------+-----------------+------------------+----------+-----------------+-------------------+----------------+---------------------+-------------------------+--------------+---------------------+--------------------------+-------------------------+------------------+----------------+--------------------+
    only showing top 5 rows
    


## Converting the label column and perserving the mappings it produced
#### We created a dictionary of each country and the label it's mapped to


```python
indexer = StringIndexer(inputCol='country_destination', outputCol='label')
train_users_raw = indexer.fit(train_users_raw).transform(train_users_raw)
label_keys = sorted(set([(i[0], i[1]) for i in train_users_raw.select(train_users_raw.country_destination, train_users_raw.label).collect()]), key=lambda x: x[0])
label_keys = {int(i[1]):i[0] for i in label_keys}
label_keys
```




    {10: 'AU',
     7: 'CA',
     8: 'DE',
     6: 'ES',
     3: 'FR',
     5: 'GB',
     4: 'IT',
     0: 'NDF',
     9: 'NL',
     11: 'PT',
     1: 'US',
     2: 'other'}



## The processed training set


```python
train_users_raw.select('features', 'label').show()
```

    +--------------------+-----+
    |            features|label|
    +--------------------+-----+
    |(75,[0,2,3,4,12,1...|  1.0|
    |(75,[0,2,3,5,12,1...|  0.0|
    |(75,[0,2,3,5,14,1...|  0.0|
    |(75,[0,2,3,11,14,...|  0.0|
    |(75,[0,3,6,13,17,...|  0.0|
    |(75,[0,3,4,12,16,...|  2.0|
    |(75,[0,1,3,4,14,1...|  0.0|
    |(75,[0,1,3,4,12,1...|  0.0|
    |(75,[0,3,5,13,16,...|  0.0|
    |(75,[0,2,3,4,14,1...|  0.0|
    |(75,[0,3,5,12,16,...|  0.0|
    |(75,[0,2,3,4,12,1...|  0.0|
    |(75,[0,2,3,5,12,1...|  0.0|
    |(75,[0,1,3,4,14,1...|  2.0|
    |(75,[0,3,4,13,16,...|  1.0|
    |(75,[0,3,5,14,17,...|  0.0|
    |(75,[0,3,4,14,17,...|  2.0|
    |(75,[0,1,2,3,4,12...|  0.0|
    |(75,[0,1,3,4,12,1...|  1.0|
    |(75,[0,3,4,12,16,...|  0.0|
    +--------------------+-----+
    only showing top 20 rows
    


## Loading and processing the test set, following the same steps done on the training set


```python
schema_Test_Users = StructType([
    StructField("id", StringType(), False),
    StructField("date_account_created", DateType(), True),
    StructField("timestamp_first_active", StringType(), True),
    StructField("date_first_booking", DateType(), True),
    StructField("gender", StringType(), True),
    StructField("age", FloatType(), True),
    StructField("signup_method", StringType(), True),
    StructField("signup_flow", FloatType(), True),
    StructField("language", StringType(), True),
    StructField("affiliate_channel", StringType(), True),
    StructField("affiliate_provider", StringType(), True),
    StructField("first_affiliate_tracked", StringType(), True),
    StructField("signup_app", StringType(), True),
    StructField("first_device_type", StringType(), True),
    StructField("first_browser", StringType(), True)])

test_users_raw = spark.read\
            .format("csv")\
            .schema(schema_Test_Users)\
            .option("header", "true")\
            .load("./Dataset/test_users.csv")
test_users_raw = test_users_raw.join(session_time, test_users_raw["id"] == session_time["user_id"],how='left_outer').select(test_users_raw["*"],session_time["sum_secs_elapsed"])
test_users_raw = test_users_raw.withColumn("month_account_created", month("date_account_created"))
test_users_raw = test_users_raw.withColumn("gender", lower(col('gender')))
test_users_raw = test_users_raw.select([column for column in test_users_raw.columns if column not in train_columnsToDrop])
test_users_raw = test_users_raw.fillna(0, subset=['age'])
test_users_raw = test_users_raw.fillna(0, subset=['sum_secs_elapsed'])
train_users_raw = train_users_raw.fillna(0, subset=['signup_flow'])
test_users_raw = test_users_raw.withColumn("age", correct_age(test_users_raw['age']))
test_users_raw = test_users_raw.withColumn("age", correct_zero_age(test_users_raw['age']))
test_users_raw = pipeline.transform(test_users_raw)
test_users_raw = test_users_raw.drop(*[column+'_indexed' for column in cat_cols])
test_users_raw = test_users_raw.withColumn("age", test_users_raw["age"].cast(FloatType()))
test_users_raw.fillna(0.0)
assembler = VectorAssembler(inputCols=[i for i in test_users_raw.columns if i not in cat_cols], outputCol='features')
test_users_raw = assembler.transform(test_users_raw)
test_users_raw.select('features').show()
```

    +--------------------+
    |            features|
    +--------------------+
    |(75,[0,1,2,3,4,12...|
    |(75,[0,2,3,4,12,1...|
    |(75,[0,1,2,3,4,12...|
    |(75,[0,2,3,4,14,1...|
    |(75,[0,1,2,3,4,12...|
    |(75,[0,2,3,4,13,1...|
    |(75,[0,2,3,4,13,1...|
    |(75,[0,2,3,4,14,1...|
    |(75,[0,2,3,5,12,1...|
    |(75,[0,1,2,3,4,13...|
    |(75,[0,1,2,3,4,12...|
    |(75,[0,2,3,5,12,1...|
    |(75,[0,2,3,4,12,1...|
    |(75,[0,2,3,4,12,1...|
    |(75,[0,2,3,4,14,1...|
    |(75,[0,1,3,4,12,1...|
    |(75,[0,2,3,4,13,1...|
    |(75,[0,2,3,8,12,1...|
    |(75,[0,1,2,3,4,13...|
    |(75,[0,2,3,4,12,1...|
    +--------------------+
    only showing top 20 rows
    


## Making a validation dataset


```python
(trainingData, validationData) = train_users_raw.randomSplit([0.8,0.2])
```

## Training a Naive Bayes model


```python
from pyspark.ml.classification import NaiveBayes
model = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = model.fit(trainingData)
validation_predictions_naive_bayes = model.transform(validationData)
validation_predictions_naive_bayes.select('prediction', 'probability').show(5)
```

    +----------+--------------------+
    |prediction|         probability|
    +----------+--------------------+
    |       0.0|[0.99863385464899...|
    |       0.0|[0.99688625623490...|
    |       4.0|[3.01118406336835...|
    |       0.0|[0.99992210079689...|
    |       4.0|[1.21004999050672...|
    +----------+--------------------+
    only showing top 5 rows
    


## Training a Decision Tree model


```python
from pyspark.ml.classification import DecisionTreeClassifier
model = DecisionTreeClassifier(labelCol="label", featuresCol="features")
model = model.fit(trainingData)
validation_predictions_decision_tree = model.transform(validationData)
validation_predictions_decision_tree.select('prediction', 'probability').show(5)
```

    +----------+--------------------+
    |prediction|         probability|
    +----------+--------------------+
    |       1.0|[0.33458177278401...|
    |       1.0|[0.33458177278401...|
    |       0.0|[0.43091482649842...|
    |       0.0|[0.56586536858600...|
    |       1.0|[0.23264861500155...|
    +----------+--------------------+
    only showing top 5 rows
    


## Training a random forest model


```python
from pyspark.ml.classification import RandomForestClassifier
model = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=3)
model = model.fit(trainingData)
validation_predictions_random_forest = model.transform(validationData)
validation_predictions_random_forest.select('prediction', 'probability').show(5)
```

    +----------+--------------------+
    |prediction|         probability|
    +----------+--------------------+
    |       0.0|[0.55331596264524...|
    |       0.0|[0.55331596264524...|
    |       0.0|[0.55331596264524...|
    |       0.0|[0.64425651464885...|
    |       0.0|[0.53348548563742...|
    +----------+--------------------+
    only showing top 5 rows
    


## Training a Logistic Regression model


```python
from pyspark.ml.classification import LogisticRegression, OneVsRest
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
ovr = OneVsRest(classifier=lr)
model = ovr.fit(trainingData)
validation_predictions_logistic = model.transform(validationData)
validation_predictions_logistic.select('prediction').show(5)
```

    +----------+
    |prediction|
    +----------+
    |       0.0|
    |       0.0|
    |       0.0|
    |       0.0|
    |       0.0|
    +----------+
    only showing top 5 rows
    


#### Logistic regression in spark does not produce a probability vector when OneVsAll wrapper is used. For this reason we will ignore it.

## Converting predictions into arrays and getting an array of true labels


```python
import numpy as np
y_predict_naive_bayes = validation_predictions_naive_bayes.toPandas().probability.apply(lambda x : np.array(x.toArray())).to_numpy()
y_predict_random_forest = validation_predictions_random_forest.toPandas().probability.apply(lambda x : np.array(x.toArray())).to_numpy()
y_predict_decision_tree = validation_predictions_decision_tree.toPandas().probability.apply(lambda x : np.array(x.toArray())).to_numpy()
y_true = [int(row.label) for row in validationData.select('label').collect()]
```

## Defining the scorer function


```python
def scorer(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    i = 0
    scores = []
    for j in range(len(y_true)):
        score = 0
        sorted_labels = np.flip(np.argsort(y_predict[j]))
        i=0
        while y_true[j] != sorted_labels[i] and i<=4:
            score += (2**(0)-1)/(np.log2(i+2))
            i+=1
        score += (2**(1)-1)/(np.log2(i+2))
        scores.append(score)
    return np.mean(scores)
```

## Comparing the scores of the three different models


```python
scorer(y_true, y_predict_naive_bayes)
```




    0.6800715061571634




```python
scorer(y_true, y_predict_decision_tree)
```




    0.8367975734716555




```python
scorer(y_true, y_predict_random_forest)
```




    0.8249663422966536



#### Based on this we choose random forest
## Recap of what we did so far:
- We loaded the datasets and cleaned them.
- We split the training dataset 80/20 to create a validation dataset.
- We created a scoriing function to evaluate the models.
- We tested four models on the validation dataset naive bayes, decision tres, random forests and logistic regression
- We chose the best performing model
- Now we will retrain the model on the entire training dataset and predict the test dataset.


```python
model = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=3)
model = model.fit(train_users_raw)
predictions = model.transform(test_users_raw)
predictions.select('prediction', 'probability').show(5)
```

    +----------+--------------------+
    |prediction|         probability|
    +----------+--------------------+
    |       0.0|[0.65797674229665...|
    |       0.0|[0.65797674229665...|
    |       0.0|[0.51901604309726...|
    |       0.0|[0.41720552566304...|
    |       0.0|[0.65797674229665...|
    +----------+--------------------+
    only showing top 5 rows
    



```python
y_predict = predictions.toPandas().probability.apply(lambda x : np.array(x.toArray())).to_numpy()
```


```python
predictions = []
for i in range(len(y_predict)):
    sorted_labels = np.flip(np.argsort(y_predict[i]))[:5]
    predictions.append([label_keys[j] for j in sorted_labels])
```

## Final Predictions


```python
predictions_df = pd.DataFrame(predictions, columns = ['prediction_'+ str(i) for i in range(1,6)])
predictions_df = pd.concat([pd.read_csv('./Dataset/test_users.csv')['id'],predictions_df],axis=1)
predictions_df.to_csv('predictions.csv')
predictions_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>prediction_1</th>
      <th>prediction_2</th>
      <th>prediction_3</th>
      <th>prediction_4</th>
      <th>prediction_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5uwns89zht</td>
      <td>NDF</td>
      <td>US</td>
      <td>other</td>
      <td>FR</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jtl0dijy2j</td>
      <td>NDF</td>
      <td>US</td>
      <td>other</td>
      <td>FR</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xx0ulgorjt</td>
      <td>NDF</td>
      <td>US</td>
      <td>other</td>
      <td>FR</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6c6puo6ix0</td>
      <td>NDF</td>
      <td>US</td>
      <td>other</td>
      <td>FR</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>czqhjk3yfe</td>
      <td>NDF</td>
      <td>US</td>
      <td>other</td>
      <td>FR</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>62091</th>
      <td>cv0na2lf5a</td>
      <td>NDF</td>
      <td>US</td>
      <td>other</td>
      <td>FR</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>62092</th>
      <td>zp8xfonng8</td>
      <td>NDF</td>
      <td>US</td>
      <td>other</td>
      <td>FR</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>62093</th>
      <td>fa6260ziny</td>
      <td>US</td>
      <td>NDF</td>
      <td>other</td>
      <td>FR</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>62094</th>
      <td>87k0fy4ugm</td>
      <td>NDF</td>
      <td>US</td>
      <td>other</td>
      <td>FR</td>
      <td>IT</td>
    </tr>
    <tr>
      <th>62095</th>
      <td>9uqfg8txu3</td>
      <td>NDF</td>
      <td>US</td>
      <td>other</td>
      <td>FR</td>
      <td>GB</td>
    </tr>
  </tbody>
</table>
<p>62096 rows × 6 columns</p>
</div>


