# Revised copy of tools.py script originally provided by Prof. Gustavo Sandoval
#
# Editor: Casey McGinley <cmm771@nyu.edu>
# Original file: https://github.com/GusSand/NyuPoly-CS6923-Fall16/blob/master/Homework%203/tools.py
# 
# New functions added by the aforementioned editor are prefaced by the 
# following: cmm771 Addition
# Original functions edited by the aforementioned editor are prefaced by the 
# following: cmm771 Edit

from collections import defaultdict
import re
import math
from unidecode import unidecode

# The default list of words to skip when counting
SKIP_WORDS_DEFAULT = ['python','data','with']

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
# cmm771 Edit
#
# Tokenize by whitespace. Use the defaultdict(int) whichsets the default 
# factory to int which makes it  the default dict useful for counting.
# 
# Take list of words to skip when counting; also skips the empty string '' 
#
def count_words(text, wc=None, skip_words=None):
    if wc == None:
        wc = defaultdict(int)
    if skip_words == None:
        skip_words = []
    tokens = text.split(" ")
    for t in tokens:
        if (t != '') and (t not in skip_words):
            wc[t] += 1  
    return(wc)

#
# cmm771 Edit
# 
# Main function. Opens the file and calls helper functions to parse
# Returns the sorted word count
# 
# Calls get_stopwords and combines the result with the skip_words list; this is
# then passed to count_words in addition to the text and word count list
#
def extract_info(filename, skip_words_manual=SKIP_WORDS_DEFAULT):
    import json
    wc = defaultdict(int)
    df = defaultdict(set)
    skip_words = get_stopwords()
    for word in skip_words_manual:
        skip_words.add(word)
    count = 0
    with open(filename) as fin:
        for line in fin:
            count += 1
            current = json.loads(line)
            text = make_text_parsable(current["abstract"] + " " + \
                current["description"] + " " + current["title"])
            wc = count_words(text, wc=wc, skip_words=skip_words)
    

    sorted_wc = sorted(wc.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_wc

# 
# cmm771 Addition
# 
# Reads in the list of stopwords from a file (filename given as param; defaults
# to stopwords.txt)
# Returns stopwords as set
#
def get_stopwords(filename='stopwords.txt'):
    stopwords = set()
    with open(filename) as sword_file:
        for line in sword_file:
            line = line.strip()
            if line == '':
                continue
            stopwords.add(line)
    return stopwords
