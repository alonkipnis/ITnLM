import pandas as pd
import numpy as np
import re

from nltk import ngrams
TOKEN_PATTERN = r"(?u)[a-zA-Z]+|\</?s\>"
SENT_START_TOKEN = '<s>'
SENT_END_TOKEN = '</s>'


def to_tokens(text):
    return re.findall(TOKEN_PATTERN, text.lower())


def normalize_text(text):
    """
    Remove/add dots to indicate sentence limits
    """
    
    text = re.sub("(Mrs?)\.", "\\1", text) # Mr/s. -> Mr/s
    text = re.sub("[!?]", ".", text) # !? -> .
    text = re.sub("(I+)\.", "\\1", text) # II. -> II
    text = re.sub("([a-zA-Z])\.([a-zA-Z])\.", "\\1\\2", text) #i.e.->ie e.g. -> eg
    return text

def add_sentence_limits(text: str, sep=r'\.') -> str:
    """
    Add SENT_START_TOKEN and SENT_END_TOKEN at the beginning and
    ending of every sentence. 
    
    Args:
        :text: is a text input
        :sep: explains how to identify sentnce ending (regular expression)
    """
    sentences = re.split(sep, normalize_text(text))
    sent_break = f' {SENT_END_TOKEN} {SENT_START_TOKEN} '
    return SENT_START_TOKEN + ' ' + sent_break.join(sentences) + ' ' + SENT_END_TOKEN


def simplify_text(text):
    """
    Returns a simplified version of the text:
     - lower case 
     - sentence limits marking
     - no punctuation or new lines
    """
    return " ".join(to_tokens(add_sentence_limits(text)))


def is_sublist(list_a: list, list_b: list)-> bool:
    """
    Is list_a a sublist of list_b?
    """
    return str(list_a).strip('[').strip(']') in str(list_b)


def ng_tokenize(text: str, ng: int) -> list:
    """
    extract ngram and add sepcial symbols
    
    Args:
      :text:  text
      :ng:    ngram level
      
    Returns:
      list of ngrams 
    """
    tokens = re.findall(TOKEN_PATTERN, text.lower())
    ngz = ngrams(tokens, ng,
                 pad_left=True, pad_right=True,
                 left_pad_symbol=SENT_START_TOKEN,
                 right_pad_symbol=SENT_END_TOKEN)
    return list(ngz)


def build_ngram_model(text: str, ng: int) -> pd.DataFrame:
    """
    1. Clean text, 
    2. Add sentence begin and end symbols
    3. Extract ngrams
    4. Remove unwanted tokens
    5. Compute frequency of every token
    
    Returns:
      dataframe. Indexes are ngrams. Columns indiacte number of occurances 
      and frequency of occurance
    """
    print("Cleaning text...")
    text = re.sub(r"<br ?/>","", text)  # remove tags <br /> and <br/>
    print("Extracting tokens...")
    tokens = ng_tokenize(text, ng)
    print("Counting tokens...")
    df_ng = pd.DataFrame(pd.DataFrame(tokens).value_counts()).rename(columns = {0 : 'count'})
    print("Computing frequencies...")
    df_ng.loc[:, 'freq'] = df_ng['count'] / df_ng['count'].sum()  # compute frequencies
    print(f"Built a model with {len(df_ng)} {ng}-grams")
    return df_ng


class State(object):
    """
    class to manage sequential state progression
    
    Args:
        past, present, future are lists
        
    Methods:
        State::step   update one step in time, so that the present 
        is appended to the past and the present gets the next value from the future
    """
    def __init__(self, past: list, present: list, future: list):
        self.past = past
        self.present = present
        self.future = future
        
    def step(self):
        self.past += self.present
        if len(self.future) > 0:
            self.present = [self.future.pop(0)]
        else:
            self.present = []
            self.future = []
            
    def print_state(self):
        print("past:", self.past)
        print("present:", self.present)
        print("future:", self.future)


def token_probability(token : str, model: pd.DataFrame) -> float:
    """
    probability of a token under the model
    
    Note: gives the marginal if number of ngrams in token is smaller
    than the size of the model
    
    If token = "" then return 1
    
    """
    if len(token) == 0:  
        return 1     # we define that an empty token has probability 1
    token_idx = tuple(token)
    
    if token_idx in model.index:
        return model.loc[token_idx].freq.sum()
    # else:     
    print(f"Unrecognized Token {token}")
    raise ValueError                
    

def conditional_probability(token_a: list, token_b: list,
                            model: pd.DataFrame, verbose=False) -> float:
    """
    Probability of token_a given token_b under the model
    (token can contain multiple words depending on the model definition)
    """
    
    pr_b = token_probability(token_b, model)
    pr_ab = token_probability(token_b + token_a, model)
    return pr_ab / pr_b
    
    
def sentence_probability(sent: str, model: pd.DataFrame,
                         verbose=False, backoff=False) -> float:
    """
    Probability of a sentence under an n-gram languge model
    
    Args:
        :sent:    the sentence 
        :model:   the model
        :verbose: flag whther to print computing process
        :bakcoff: try backing off to handle unknown ngrams
        
    Returns:
       probability
    """
    
    ng = len(model.index[0])  # identify model order

    sent_atoms = sent.split()  
    first_token = sent_atoms[:1] 

    word_stream = State(past=[], present=first_token, future=sent_atoms[1:])

    # update state
    logprob = 0
    while len(word_stream.present) > 0:
        if backoff:
            pr_token = conditional_probability_backoff(word_stream.present,
                                                       word_stream.past[-ng+1:],
                                                       model, verbose=verbose)
        else:
            pr_token = conditional_probability(word_stream.present, word_stream.past[-ng+1:],
                                               model)
        logprob += np.log(pr_token)
        if verbose:
            word_stream.print_state()
            print(f"P(present|past) = {pr_token}")
            print("------------------------------------")
        word_stream.step()

    return np.exp(logprob)
