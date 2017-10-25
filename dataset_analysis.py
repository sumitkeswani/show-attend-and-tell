mport string
from nltk.stem.porter import *
import json

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

	'''
	function to preprocess captions and create a mapping from image ID to a collection of words present in its captions

	structure of caption mapping:
	{
	image_id1: set(word1, word2 ....),
	image_id2: set(word1, word2 ....),
	 .
	 .
	 .
	 .
	image_idn: set(word1, word2 ....) }
	'''
	cmapping = {}

	# loading the JSON
	with open(path) as caption_file:
		caps = json.load(caption_file)

	# getting each caption, processing it, 
	# then storing it as words for that particular image ID
	for caption_data in caps["annotations"]:
		image_id = caption_data["image_id"]
		processed_caption = _process_text(caption_data["caption"].encode("utf-8"))
		if image_id not in cmapping:
			cmapping[image_id] = processed_caption.split()
		else:
			cmapping[image_id].extend(processed_caption.split())

	# make set of each list of words in cmapping
	for key in cmapping.keys():
		cmapping[key] = set(cmapping[key])

	return cmapping

def _preprocess_question(question_path, answer_path):
	# answer_path = "../../answers/training/mscoco_train2014_annotations.json"
	# question_path = "./OpenEnded_mscoco_train2014_questions.json"

	'''
	1. function to preprocess questions and create a mapping from
	image ID to a collection of words present in its questions
	2. Exclude questions whose answers are no
	structure of question mapping:
	{
	image_id1: set(word1, word2 ....),
	image_id2: set(word1, word2 ....),
	 . 
	 .
	 .
	 .
	image_idn: set(word1, word2 ....) }
	'''
	qmapping = {}
	with open(question_path, "r") as question_file:
		ques = json.load(question_file)
	with open(answer_path, "r") as answer_file:
		ans = json.load(answer_file)
	# generate a mapping for question_id to answer index
	# from answer json file
	qa_mapping = {}
	for i, answer_data in enumerate(ans["annotations"]):
		qa_mapping[answer_data["question_id"]] = i
	#initialize qmapping
	for question_data in ques["questions"]:
		qmapping[question_data["image_id"]] = []
	# getting each question, processing it,
	# then storing it as words for particular image ID
	for question_data in ques["questions"]:
		answer_index = qa_mapping[question_data["question_id"]]
		if(ans["annotations"][answer_index]["multiple_choice_answer"] == "yes"):
			processed_question = _process_text(question_data["question"].encode("utf-8"))
			qmapping[question_data["image_id"]].extend(processed_question.split())
	# creating set from lists in qmapping
	for key in qmapping.keys():
		qmapping[key] = set(qmapping[key])

	return qmapping

def caption_question_match(caption_path, question_path):

	#measuring the number of matching words in its' captions and questions for every image ID
	matching_words = {}
	cmapping = _preprocess_caption(caption_path)
	qmapping = _preprocess_question(question_path)
	#preprocessing captions and questions by passing the path

	for qid in qmapping.keys():
		if qid in cmapping.keys():
			#joining both mappings for a particular ID, taking a unique set/intersection of the join and then finding the length
			matching_words_count = len(qmapping[qid]).intersection(cmapping[qid])
			matching_words[qid] = {"matching_words_count": matching_words_count, "question_length": len(qmapping[qid]), "caption_length": len(cmapping[qid])}

	return matching_words

def main():

	caption_path = "../datasets/COCO-VQA/captions/validation/annotations/captions_val2014.json"
	question_path = "../datasets/COCO-VQA/questions/validation/OpenEnded_mscoco_val2014_questions.json"

	#Data Analysis

	# Overlap in number of common words between caption and question for every image ID
	matching_words_count = caption_question_match(caption_path, question_path)

	match = open('data_analysis.json', 'w+')
	match.write(json.dumps(json.loads(matching_words_count),sort_keys=True, indent = 4))
	match.close()

if __name__ == '__main__':
	main()


