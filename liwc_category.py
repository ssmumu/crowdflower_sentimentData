##########################################################
##														##			
##				Author: Samiha Samrose					##							
##				samiha.mumu@gmail.com					##
##														##	
##########################################################


import pandas as pd
import numpy as np
from liwc_example import Liwc
import matplotlib.pyplot as plt
import random

liwc = Liwc('./liwcdic2007.dic','./liwccat2007.txt')
mainCategory = ['Social', 'Affective', 'Cognitive', 'Perceptual', 'Biological', 'Relativity']
target_names = ['empty','sadness','enthusiasm','neutral','worry','surprise','love','fun','hate','happiness','boredom','relief','anger']


def readTweetProperty(filename,feature_col, feature, tweet):
	df = pd.read_csv(filename)
	values = ""
	for i in range(0, len(df.index)):
		if df[feature_col][i] == feature:
			values += ((str(df[tweet][i]) + " "))
	return values

def testLiwcMainCat(counts_d, category):
	labels = []
	sizes = []
	for key,val in counts_d.items():
		if liwc.catdic[key] in category:
			print(liwc.catdic[key], val)
			labels.insert(len(labels), liwc.catdic[key])
			sizes.insert(len(sizes), val)
	return labels, sizes

def plotPie(labels, sizes, title):
	colors = []
	for i in range(0,30):
		colors.insert(len(colors),'#{:06x}'.format(random.randint(0, 250**3)))
	plt.title(title)
	plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=200) 
	centre_circle = plt.Circle((0,0),0.55,color='black', fc='white',linewidth=1.25)
	fig = plt.gcf()
	fig.gca().add_artist(centre_circle)
	plt.axis('equal') 
	plt.savefig(title+'.png', bbox_inches='tight')
	plt.clf()


def liwcStat(ttext, t):  
	title = t+" - Characteristics of Words in Tweets"
	print title
	words_list = ttext.split()
	counts_d = liwc.get_counts_dict(words_list)
	labels, sizes = testLiwcMainCat(counts_d, mainCategory)
	plotPie(labels, sizes, title)


for t in target_names:
	ttext = readTweetProperty('text_emotion.csv','sentiment',t, 'content')
	liwcStat(ttext, t)
