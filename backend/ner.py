# import all from ner_model.py
from ner_model import *
# import medmon.py
from medmon import Medmon

# get news data from medmon
mdmon = Medmon()
# get dataframe news
df_medmon = mdmon.db_connect()

def cleaningSentence():
    return 0

# =======  Predict untuk sentence manual input ==========
def predictNer(sentence="TPUA Desak Jokowi Mundur, TB Hasanuddin: Jangan Halu, Mendesak Presiden Mundur Bukan Perkara Mudah"):
    words, infer_tags, unknown_tokens = trainer.infer(sentence=sentence)

    ner_dict = dict(zip(words, infer_tags)) 
    return ner_dict

# =======  Predict berita secara otomatis ==========
def predictAll(df=df_medmon, predictNer=predictNer):
    for index, row in df.iterrows():
        ner_dict_result = predictNer(row['news_content'])
        print(ner_dict_result)


predictAll()