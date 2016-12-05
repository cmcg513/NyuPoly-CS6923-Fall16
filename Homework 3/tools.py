from collections import defaultdict
import re
import math
from unidecode import unidecode

spchars = re.compile('\`|\~|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\_|\+|\=|\\|\||\{|\[|\]|\}|\:|\;|\'|\"|\<|\,|\>|\?|\/|\.|\-')

# Utility function that does the following to the text:
# - Convert to unicode
# - Convert to lowercase
# - Remove special chars
def make_text_parsable(text):
    # convert to unicode
    text = unidecode(text) #.decode('utf-8', 'ignore'))
    # convert text to lowercase
    text = text.lower()
    # remove special characters
    text = spchars.sub(" ", text)
    return(text)

#
# Tokenize by whitespace. Use the defaultdict(int) whichsets the default 
# factory to int which makes it  the default dict useful for counting. 
#
# def count_words(text, wc=None):
#     if wc == None:
#         wc = defaultdict(int)
#     tokens = text.split(" ")
#     for t in tokens:
#         wc[t] += 1  
#     return(wc)
def get_tf(text,u_words):
    tf = defaultdict(int)
    tokens = text.split(" ")
    for t in tokens:
        tf[t] += 1
        u_words.add(t)
    return tf,u_words

def get_df(u_words,tf_tables):
    df = defaultdict(int)
    for word in u_words:
        for tf in tf_tables:
            if tf[word] != 0:
                df[word] += 1
    return df

def get_idf(u_words,n_docs,df):
    idf = defaultdict(float)
    for word in u_words:
        idf[word] = math.log(float(n_docs)/float(df[word]))
    return idf

def get_max_tfidf(u_words,tf_tables,idf):
    max_tfidf = defaultdict(float)
    for word in u_words:
        max_score = 0
        for tf in tf_tables:
            tfidf = float(tf[word]) * idf[word]
            if tfidf > max_score:
                max_score = tfidf
        max_tfidf[word] = max_score
    return max_tfidf

#
# Main function. Opens the file and calls helper functions to parse
# Returns the sorted word count
#
def extract_info(filename):
    import json
    tf_tables = []
    u_words = set()
    count = 0
    with open(filename) as fin:
        for line in fin:
            count += 1
            current = json.loads(line)
            text = make_text_parsable(current["abstract"] + " " + \
                current["description"] + " " + current["title"])
            # wc = count_words(text, wc)
            tf,u_words = get_tf(text,u_words)
            tf_tables.append(tf)
    df = get_df(u_words,tf_tables)
    idf = get_idf(u_words,count,df)
    max_tfidf = get_max_tfidf(u_words,tf_tables,idf)
    sorted_tfidf = sorted(max_tfidf.items(), key=lambda x: x[1], reverse=True)
    
    # import IPython; IPython.embed()

    return sorted_tfidf

def main():
    tfidf = extract_info('data_file.txt')
    print("Top ten words by TF*IDF:\n")
    for i in range(0,10):
        print(str(i+1)+". "+tfidf[i][0]+" - "+str(tfidf[i][1]))

main()