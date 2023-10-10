from collections import Counter
class Vocab:

    def __init__(self, df, source=None):
        self.word2index = {}
        self.index2word = {}
        self.word_vocab = []
        self.text = []
        '''
        Gathers text from contexts and questions to build a vocabulary.

        :param df: dataframe of SQUAD dataset.
        :returns: list of contexts and questions
        '''

        total = 0
        if source is None:
            unique_questions = list(df.Question.unique())
            unique_answers = list(df.Answer.unique())
            total += df.Answer.nunique() + df.Question.nunique()
            self.text.extend(unique_questions + unique_answers)
        elif source:
            unique_questions = list(df.Question.unique())
            total += df.Question.nunique()
            self.text.extend(unique_questions)
        else:
            unique_answers = list(df.Answer.unique())
            total += df.Answer.nunique()
            self.text.extend(unique_answers)

    def __call__(self, tokens):
        indices = [self.__getitem__(word) for word in tokens]
        return indices
    
    def __getitem__(self, word):
        try:
            return self.word2index[word]
        except KeyError:
            return 0
        
    def __len__(self):
        return len(self.word_vocab)
    
    def build_word_vocab(self):
        '''
        Builds a word-level vocabulary from the given text.

        :param list text: list of questions/answers
        :returns 
            dict word2idx: word to index mapping of words
            dict idx2word: integer to word mapping
            list word_vocab: list of words sorted by frequency
        '''


        words = []
        for sent in self.text:
            for word in sent.split(" "):
                words.append(word)

        word_counter = Counter(words)
        self.word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
        print(f"raw-vocab: {len(self.word_vocab)}")
        # self.word_vocab.insert(0, '<unk>')
        self.word_vocab.insert(1-1, '<pad>')
        self.word_vocab.insert(2-1, '<sos>')
        self.word_vocab.insert(3-1, '<eos>')
        print(f"vocab-length: {len(self.word_vocab)}")
        self.word2index = {word:idx for idx, word in enumerate(self.word_vocab)}
        print(f"word2idx-length: {len(self.word2index)}")
        self.index2word = {v:k for k,v in self.word2index.items()}
        
    