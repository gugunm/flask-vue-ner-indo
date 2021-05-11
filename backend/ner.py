# import all from ner_model.py
from ner_model import *
# import medmon.py
from medmon import Medmon
import re
import html


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleanfromhtml = re.sub(cleanr, '', raw_html)

  # clean from entities html
  cleantext = html.unescape(cleanfromhtml)
  return cleantext


# =======  Predict untuk sentence manual input ==========
def predictNerOfSentence(sentence="TPUA Desak Jokowi Mundur, TB Hasanuddin: Jangan Halu, Mendesak Presiden Mundur Bukan Perkara Mudah"):
    words, infer_tags, unknown_tokens = trainer.infer(sentence=sentence)

    ner_dict = dict(zip(words, infer_tags)) 
    return ner_dict

# =======  Predict berita secara otomatis ==========
def predictByDate(
        predictNerOfSentence=predictNerOfSentence,
        tahun=2021,
        bulan=5,
        tgl=1,
        limit=3
    ):
    # array of dictionary news_content
    list_ner_dict = []
    # get news data from medmon
    mdmon = Medmon()
    # get dataframe news
    df_medmon = mdmon.db_connect(
        tahun=tahun,
        bulan=bulan,
        tgl=tgl,
        limit=limit
    )

    for index, row in df_medmon.iterrows():
        # get news content and clean it 
        row['news_content'] = cleanhtml(row['news_content'])
        # predict ner of news content
        ner_dict_result = predictNerOfSentence(row['news_content'])
        # create a json
        list_ner_dict.append(ner_dict_result)
        # print(ner_dict_result)
        # break

    # create json for its date of pubdate
    news_json_by_pubdate = {
        "pubdate": "{}-{}-{}".format(tahun, bulan, tgl),
        "news": list_ner_dict
    }

    # print(news_json_by_pubdate)

    return news_json_by_pubdate


predictByDate()