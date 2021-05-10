import mysql.connector as connection
import pandas as pd
from decouple import config
from datetime import datetime

class Medmon:
    def __init__(self):
        self.host= config('DB_HOST')
        self.database =config('DB_NAME')
        self.user=config('DB_USERNAME')
        self.passwd=config('DB_PASSWORD')
        self.db_dataframe=""
    
    def db_connect(self):
        # str_query = "SELECT * from data_news \
        #         WHERE news_pubday='2021-04-19' \
        #         ORDER BY news_pubdate DESC LIMIT 100;"

        str_query = "\
            SELECT * from data_news \
                WHERE news_pubday='{}' \
                    ORDER BY news_pubdate DESC LIMIT 3 \
                        ;".format(datetime.today().strftime("%Y-%m-%d"))
        
        try:
            db_medmon = connection.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                passwd=self.passwd,
                use_pure=True
                )
            self.db_dataframe = pd.read_sql(str_query, db_medmon)
            db_medmon.close()

            return self.db_dataframe
        except Exception as e:
            return str(e)

# medmon = Medmon()
# print(medmon.host)
# medmon.db_connect()

# print(medmon.db_dataframe)



# df = result_dataFrame.copy()
# news_title = df.news_title.values
# print(news_title[4])
