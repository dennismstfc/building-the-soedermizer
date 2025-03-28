import pandas as pd


class Sentence:
    """
    The sentence class is used for the dataset creation for Goal 1. 
    """
    def __init__(self, row: dict):
        """
        :param row: A dictionary containing the sentence in multiple versions, i.e., maskulinum, femininum, 
        neutral, doppelpunkt, unterstrich, sternchen, schraegstrich, grossbuchstaben.
        """
        # Non-gendered sentences
        maskulinum = row["maskulinum"]
        femininum = row["femininum"]
        self.non_gendered = [maskulinum, femininum]

        # Gendered
        neutral = row["neutral"]
        doppelpunkt = row["inkl_Doppelpunkt"]
        unterstrich = row["inkl_Unterstrich"]
        sternchen = row["inkl_Sternchen"]
        schraegstrich = row["inkl_Schrägstrich"]
        grossbuchstaben = row["inkl_Grossbuchstaben"]
        self.gendered = [neutral, doppelpunkt, unterstrich, sternchen, schraegstrich, grossbuchstaben]

    def subset_all_combinations(self) -> pd.DataFrame:
        """
        :return: A DataFrame containing all possible combinations of gendered and non-gendered sentences.
        (maskulinum, femininum) x (neutral, doppelpunkt, unterstrich, sternchen, schraegstrich, grossbuchstaben)
        """
        df = pd.DataFrame()

        gendered = []
        non_gendered = []

        for gendered_sentence in self.gendered:
            for non_gendered_sentence in self.non_gendered:
                gendered.append(gendered_sentence)
                non_gendered.append(non_gendered_sentence)
        
        df["gendered"] = gendered
        df["non_gendered"] = non_gendered

        return df
    
    @property
    def gendered_sentences(self) -> list:
        """
        :return: A list of all gendered sentences.
        """
        return self.gendered
    
    @property
    def male(self) -> str:
        """
        :return: The male version of the sentence.
        """
        return self.non_gendered[0]
    
    @property
    def female(self):
        """
        :return: The female version of the sentence.
        """
        return self.non_gendered[1]


class SentenceEvaluator:
    """
    Class that is designed for the evaluation of sentences. It is used to compare two sentences
    and to check if they are equal or not.
    """
    def __init__(self, sentence: str) -> None:
        sentence_raw = sentence.strip()
        self.sentence = sentence_raw.replace("\n", " ")
        
        lower = self.sentence.lower()
        self.lower = lower.split()


    def __sub__(self, other) -> int:
        """
        Checks the difference between two sentences. Diff is calculated as the symmetric
        difference between the two sentences. The substraction operator returns the length 
        of the difference. If the difference is 2, then most likely another gender is used 
        in the prediction sentence. E.g. "Der Bäcker hat gebacken" (true value) vs. "Die Bäckerin
        hat gebacken" (predicted value) would return 2. Since the translation task was designed 
        to not differentiate between the maskulin and feminin form, a difference of 2 is acceptable.
        :param other: Another Sentence object
        """
        sentence_a = self.lower
        sentence_b = other.lower

        diff = set(sentence_a) ^ set(sentence_b)
        return len(diff)
    
    def get_difference(self, other) -> set:
        """
        Checks the difference between two sentences. 
        :param other: Another Sentence object
        """
        sentence_a = self.lower
        sentence_b = other.lower

        return set(sentence_a) ^ set(sentence_b)

    def __eq__(self, other):
        """
        Checks if two sentences are equal.
        :param other: Another Sentence object
        """
        return self.lower == other.lower