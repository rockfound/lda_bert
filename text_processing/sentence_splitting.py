import re

# Expressions for the split into sentences regex based function below.
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|ft|cu|CU|Cu|vs)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
conjs = ['but', 'however', 'except', 'although', 'otherwise',
         'with the exception', 'the only problem', 'other than']


def split_into_sentences(text):
    """This function takes a string and splits it into it's component sentences
        works by splitting on punctuation with carefully chosen exceptions.
        Not written by James D., found here originally and tweaked for our uses:
    https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
    """
    for x in [w for w in text.split() if
              w[:-1].replace('.', '', 1).replace('$', '', 1).replace('"', '', 1).replace(',', '').isdigit()]:
        text = text.replace(x, '<prd>'.join(x.split('.')), 1)

    text = " " + str(text) + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    if "e.g." in text: text = text.replace("e.g.", "for example")
    text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]",
                  "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return [x for x in sentences if len(x) > 1]
