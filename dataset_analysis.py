import json
import string
from nltk.stem.porter import *

def _process_text(text):

	'''Process the text. 
	Input: string - can either be a caption or question
	Output: string - preprocessed string
	Following steps are performed:
	1. all small
	2. Whitespace removal
	3. Removal of stop stop_words
	4. & changes to 'and'
	5. Removing puctuations
	6. lammetization of words
	'''
	#List of stopWords obtained from: https://www.ranks.nl/stopwords
	stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]

	# all small text
	text = text.lower()

	# Whitespace to ' '
	text = " ".join(text.split())

	# Removing stop words
	temp = [word for word in text.split() if word not in stop_words]
	text = " ".join(temp)

	# & -> 'and'
	text = text.replace("&", " and ")
	text = " ".join(text.split())

	# remove punctuations
	text = text.translate(None, string.punctuation)

	# stemming each word
	stemmer = PorterStemmer()
	stemmed_words = [stemmer.stem(word) for word in text.split()]
	text = " ".join(stemmed_words).encode("utf-8")

	return text

def _preprocess_caption(path):

	#function to preprocess captions and create a mapping from image ID to a collection of words present in its captions

	#structure of caption mapping:
	#{
	# image_id1: [word1, word2 ....],
	# image_id2: [word1, word2 ....],
	#  .
	#  .
	#  .
	#  .
	# image_idn: [word1, word2 ....]
	#}

	cmapping = {}
	#opening the JSON
	with open(path) as captionfile:
		caps = json.dump(captionfile, indent = 4)

	#getting each caption, processing it, then storing it as words for that particular image ID
	for key in caps["annotations"]:
		capstring = _process_text(key["caption"].encode('utf8'))
		cmapping[key["image_id"]] = set(cmapping[key["image_id"]].append(capstring.split()))

	return cmapping

def _preprocess_question(path):

	#function to preprocess questions and create a mapping from image ID to a collection of words present in its questions

	#structure of question mapping:
	#{
	# image_id1: [word1, word2 ....],
	# image_id2: [word1, word2 ....],
	#  .
	#  .
	#  .
	#  .
	# image_idn: [word1, word2 ....]
	#}

	qmapping = {}
	with open(path) as questionfile:
		ques = json.dump(questionfile, indent = 4)


	#getting each question, processing it, then storing it as words for that particular image ID
	for key in ques["questions"]:
		qstring = _process_text(key["question"].encode('utf8'))
		qmapping[key["image_id"]] = set(qmapping[key["image_id"]].append(qstring.split()))

	return qmapping

text = "This is super amaing! Really excited about it.. Just wait & watch!!"
ans = _process_text(text.encode("utf-8"))
print ans


