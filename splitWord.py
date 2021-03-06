from ckiptagger import data_utils, construct_dictionary, WS
def wordSplit(results):
    Ws = WS("./data")
    #Pos = POS("./data")
   # Ner = NER("./data")

    sentence_list = [
        results,
        
    ]

    word_sentence_list = Ws(
        sentence_list,
        # sentence_segmentation = True, # To consider delimiters
        # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
        # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
        # coerce_dictionary = dictionary2, # words in this dictionary are forced
    )

    #pos_sentence_list = Pos(word_sentence_list)

    #entity_sentence_list = Ner(word_sentence_list, pos_sentence_list)
    #print("詞性的list:")
    #print(pos_sentence_list)
    print("斷詞的list:")
    print(word_sentence_list)
    return word_sentence_list