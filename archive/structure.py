from typing import List

class WordPair:
    """
    Using a struct-like class to store the data of a word pair and do some processing on it.
    """
    def __init__(self, word_dict: dict):
        self.__non_gendered = word_dict["non_gendered"]
        self.__gendered = word_dict["gendered"]
    
    @property
    def non_gendered(self) -> str:
        # Remove the text in brackets from the non-gendered word
        tmp_word = self.__non_gendered.split(" (")[0]
        tmp_word = tmp_word.split(" [")[0]
        return tmp_word.strip()

    @property
    def gendered(self) -> List[str]:
        processed_words = []

        for act_word in self.__gendered:
            tmp_word = act_word.split(" (")[0]
            tmp_word = tmp_word.split("...")[0]
            processed_words.append(tmp_word.strip())

        # Remove empty strings
        processed_words = list(filter(None, processed_words))
        return processed_words

    @property
    def is_plural(self) -> bool:
        # Check if the non-gendered word is plural (it has in brackets "pl.")
        return "(pl.)" in self.__non_gendered

    @property
    def is_singular(self) -> bool:
        # Check if the non-gendered word is singular (it has in brackets "sg.")
        return "(sg.)" in self.__non_gendered
    
    @property
    def is_plural_and_singular(self) -> bool:
        # Check if the non-gendered word has both singular and plural forms. i.e. "(sg./pl.)"
        return "(sg./pl.)" in self.__non_gendered
    
    @property
    def no_formulation_found(self) -> bool:
        # Iterate through the gendered words and check if "kein passender Begriff" is present
        for act_word in self.__gendered: 
            if "kein passender Begriff" in act_word:
                return True
        return False

    @property
    def is_already_gender_neutral(self) -> bool:
        # Iterate through the gendered words and check if "bereits genderneutral" is present
        for act_word in self.__gendered: 
            if "bereits genderneutral" in act_word:
                return True
            if "schon gendergerecht" in act_word:
                return True
            if "bereits gendergerecht" in act_word:
                return True

        return False


# Example usage
if __name__ == "__main__": 
    word_data = {
        "non_gendered": "Student [m]",
        "gendered": ["Student (m)", "Studentin (f)", "Studierende (m/f)", "Test...", "Text ..."]
    }

    word_pair = WordPair(word_data)

    print(word_pair.non_gendered)
    print(word_pair.gendered)