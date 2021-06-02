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

    # ner_dict = dict(zip(words, infer_tags)) 
    # return ner_dict
    arr_result = []
    for i, word in enumerate(words):
        arr_result.append(
            {
                "word": word,
                "tag": infer_tags[i]
            }
        )
    return arr_result

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
        tahun=int(tahun),
        bulan=int(bulan),
        tgl=int(tgl),
        limit=limit
    )

    for index, row in df_medmon.iterrows():
        # get news content and clean it 
        row['news_content'] = cleanhtml(row['news_content'])
        row['news_title'] = cleanhtml(row['news_title'])
        # predict ner of news content
        ner_content_result = predictNerOfSentence(row['news_content'])
        ner_title_result = predictNerOfSentence(row['news_title'])
        # create a json
        list_ner_dict.append(
            {
                "title": row['news_title'],
                "content": row['news_content'],
                "ner_title": ner_title_result,
                "ner_content": ner_content_result,
                "media": row['news_media'],
                "pubday": row['news_pubday'],
                "url": row['news_url']
            }
        )
        # print(ner_dict_result)
        # break

    # create json for its date of pubdate
    news_json_by_pubdate = {
        # "pubdate": "{}-{}-{}".format(tahun, bulan, tgl),
        "news": list_ner_dict
    }

    # print(news_json_by_pubdate)

    return news_json_by_pubdate


predictByDate()