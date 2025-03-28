import json

from standard_values import *
from scraping import scrape_gendered_words
from structure import WordPair

class GenderDictionary:
    """
    Class for loading and processing the scraped gendered words data. The data is scraped from the
    website geschicktgendern.de and contains a list of words with their gendered
    counterparts. The class also provides a method to replace non-gendered words in a sentence with their 
    gendered counterparts.    
    """

    def __init__(self):
        self.__load_data()

        # Removing wrong characters from the alphabet
        self.alphabet = list(self.raw_data.keys())
        self.alphabet.remove('(')
        self.alphabet.remove('[')
        self.alphabet.remove('.')

        self.__process_data()
        self.__refine_dictionary()

    def __load_data(self) -> None:
        # Check if the raw data file exists
        if not RAW_DATA_PATH.exists():
            print("Cannot find existing data file. Scraping data...")
            scrape_gendered_words()
        else:
            print("Found scraped data. Loading it...")

        with open(RAW_DATA_PATH, 'r', encoding='utf-8') as json_file:
            self.raw_data = json.load(json_file)

    def __process_data(self) -> None:
        """
        Processes the raw data and creates a new dictionary with the same structure, but with the
        WordPair objects instead of the raw data. This is an intermediate step before refining the        
        dictionary and is used for debugging purposes. 
        """
        processed_data = {}

        for letter in self.alphabet:
            processed_letter = {}

            act_letter = self.raw_data[letter]
            word_keys = list(act_letter.keys())
            word_keys.pop(0) # Remove the first key, which is the letter itself

            for word in word_keys:
                word_data = act_letter[word]
                word_pair = WordPair(word_data)

                processed_letter[word] = {
                    "non_gendered": word_pair.non_gendered,
                    "gendered": word_pair.gendered,
                    "no_formulation_found": word_pair.no_formulation_found,
                    "already_gender_neutral": word_pair.is_already_gender_neutral,
                    "is_plural": word_pair.is_plural,
                    "is_singular": word_pair.is_singular,
                    "is_plural_and_singular": word_pair.is_plural_and_singular
                }
            
            processed_data[letter] = processed_letter        

        if not PROCESSED_DATA_PATH.exists():
            PROCESSED_DATA_PATH.touch()

        with open(PROCESSED_DATA_PATH, 'w', encoding='utf-8') as json_file:
            json.dump(processed_data, json_file, ensure_ascii=False, indent=4)
    
    def __refine_dictionary(self) -> None:
        """
        Refines the processed data by removing all entries where no formulation was found and
        where the word is already gender neutral. This is the final dictionary that will be used
        for the gendering process. Also the structure of the dictionary is changed to a flat
        dictionary with the word as the key and the WordPair object as the value.
        """
        # Load the processed data
        with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as json_file:
            self.processed_data = json.load(json_file)
        
        # Create the final dictionary
        final_dictionary = {}

        for letter in self.alphabet:
            act_letter = self.processed_data[letter]
            word_keys = list(act_letter.keys())

            for word in word_keys:
                word_data = act_letter[word]

                if not word_data["no_formulation_found"]:
                    if word_data["already_gender_neutral"]:
                        continue
                
                    word_pair = WordPair(word_data)

                    # Add the word pair to the final dictionary and check if it is singular or plural.
                    # There are problably irregular cases where this does not work, but it is a good
                    # starting point.
                    if word_data["is_singular"]:
                        final_dictionary["der " + word_pair.non_gendered] = word_pair.gendered
                    elif word_data["is_plural"]:
                        final_dictionary["die " + word_pair.non_gendered] = word_pair.gendered
                    else:
                        final_dictionary[word_pair.non_gendered]  = word_pair.gendered
                
        # Selecting some words to remove from the dictionary by hand
        to_remove = ["man"]

        for word in to_remove:
            if word in final_dictionary:
                del final_dictionary[word] 

        if not FINAL_DICTIONARY_PATH.exists():
            FINAL_DICTIONARY_PATH.touch()

        with open(FINAL_DICTIONARY_PATH, 'w', encoding='utf-8') as json_file:
            json.dump(final_dictionary, json_file, ensure_ascii=False, indent=4)

    def translate(self, sentence: str) -> str:
        """
        Takes an arbitrary sentence and replaces all non-gendered words with their gendered counterparts
        using the final dictionary created in __refine_dictionary().
        :param sentence: The sentence to process (str)
        :return: The processed sentence (str)
        """
        # Make sure the final dictionary exists
        if not FINAL_DICTIONARY_PATH.exists():
            self.__refine_dictionary()
        
        # Load the final dictionary
        with open(FINAL_DICTIONARY_PATH, 'r', encoding='utf-8') as json_file:
            self.final_dictionary = json.load(json_file)

        # Replace all non-gendered words with their gendered counterparts
        # Note: This is a very naive approach and will not work for all cases
        for word in sentence.split():
            if word in self.final_dictionary:
                # There are cases where there are more gendered versions of a word  
                # in the dictionary. We will just use the shortest one for now.
                translations =  self.final_dictionary[word]
                sentence = sentence.replace(word, min(translations, key=len))
        
        return sentence


if __name__ == "__main__":
    gender_dict = GenderDictionary()

    sentence = "Der Lehrer ist cool."
    print("Original sentence:", sentence)
    print("Processed sentence:", gender_dict.translate(sentence))