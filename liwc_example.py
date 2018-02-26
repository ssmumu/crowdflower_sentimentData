#!/usr/bin/python

from __future__ import print_function
import re
from collections import defaultdict
import numpy as np
import io

#-------------------------------------------------------------------------------
class Liwc:
    """ 
    This class will allow you to count the occurences of word categories from a 
    given text. Is configured to use the LIWC 2007 dictionary and catogories.
    """
    def __init__(s, dic_file, cat_file=None):
        s.dic = {}
        s.catdic = {}
        s.ReadLIWCDictionary(dic_file)
        if(cat_file):
            s.ReadLIWCCategories(cat_file)

    #---------------------------------------------------------------------------
    def get_cats(s, word):
        """ Returns list of categories associated with word. 
            If no match, tries to find matches in wildcard substrings.
        """
        if word in s.dic:
            return s.dic[word]
        
        for i in range(1,len(word)):
            key = word[:i] + "*"
            if key in s.dic:
                return s.dic[key]
            
        return [-1]
    
    #---------------------------------------------------------------------------
    def get_counts_dict(s,words_list):
        """ given a list of words, counts occurences for each category and 
            returns as a dict """
        
        words_list = [w.lower() for w in words_list]
        counts = defaultdict(int)        
        for word in words_list:
            word_categories = s.get_cats(word)
            for cat in word_categories:
                counts[cat] += 1
        
        return counts

    #---------------------------------------------------------------------------
    def get_counts_a(s,words_list):
        """ given a list of words, counts occurences for each category and 
            returns as a np array """
        
        words_list = [w.lower() for w in words_list]
        counts_a = np.zeros(len(s.catdic),dtype=int)        
        cat_names = np.empty_like(counts_a,dtype=object)        
        counts_d = s.get_counts_dict(words_list)
        
        i = 0
        for cat,name in sorted(s.catdic.items()):
            cat_names[i] = name
            counts_a[i] = counts_d[cat]
            i += 1
        
        return counts_a
    
    #---------------------------------------------------------------------------
    def ReadLIWCDictionary(s, dic_file):
        """ Loads dic_file into s.dic. 
            Each line of the dict file is parsed, first token is the word or
            word root followed by the category indices.
        """
        f = io.open(dic_file, encoding='iso-8859-15')
        lines = f.readlines()
        f.close()
        
        s.dic = {}
    
        for line in lines:
            tokens = line.lstrip().rstrip().split("\t")
            word = tokens[0]
            categories = set([int(cat) for cat in tokens[1:]])
            s.dic[word] = categories
    
    
    #---------------------------------------------------------------------------
    def ReadLIWCCategories(s,path):
        f = open(path)
        lines = f.readlines()
        f.close()
        #categories = lines[0].split("\r")
        s.catdic = {}
        
        #for cat in categories:
        for cat in lines:
            catparts = cat.split()
            s.catdic[int(catparts[0])] = ' '.join(catparts[1:])
            
        s.catdic[-1] = 'NOT IN DICT'
            

###########################################################################
#--------------------------------------------------------------------------
def example_1(liwc):
    print('\nExample 1: getting categories for a word\n')
    
    word = "like"
    cats = liwc.get_cats(word)
    print('WORD: ', word)
    print('CATEROGORY NAMES:', [liwc.catdic[x] for x in cats])

#--------------------------------------------------------------------------
def example_2(liwc):
    print('\nExample 2: getting counts from text\n')
    text = 'I like to eat chicken because chicken is my favorite food I love meat'
    print('TEXT: ',text)
    
    words_list = text.split()
   
    counts_d = liwc.get_counts_dict(words_list)
    
    print('\nUsing get_counts_dict')
    for key,val in counts_d.items():
        print(liwc.catdic[key],':',val)



    print('\n\n########################\n\nUsing get_counts_a:')    
    counts_a = liwc.get_counts_a(words_list)
    count_names = [name for cat,name in sorted(liwc.catdic.items())]

    for i,name in enumerate(count_names):
        print(name + ':',counts_a[i])    

    
#--------------------------------------------------------------------------
# def main():
#     liwc = Liwc('./liwcdic2007.dic','./liwccat2007.txt')
    
#     example_1(liwc)
#     example_2(liwc)

# ###########################################################################
# if __name__ == "__main__":
#     main()