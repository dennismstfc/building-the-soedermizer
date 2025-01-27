import pandas as pd

class Sentence:
    """
    The sentence class to handle the different versions of a sentence. 
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