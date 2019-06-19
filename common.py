import os
import numpy as np
import pandas as pd  # noqa
import matplotlib.pyplot as plt  # noqa
import spacy  # noqa
from spacy import displacy
from ipywidgets import interact, Dropdown, SelectionSlider, HBox, VBox
import ipywidgets as widgets
from IPython.display import clear_output, display
from ast import literal_eval


class Tree():
    """Simple tree class for easy manipulation of the dependency trees

    e.g. Tree({'text': 'is', 'idx': 1},
             [Tree({'text': 'He', 'idx': 0}),
              Tree({'text': 'here', 'idx': 2})])
    """

    def __init__(self, attributes, children=[]):
        self.attributes = attributes
        self.children = children
        for child in self.children:
            child.parent = self

    def __repr__(self):
        txt = self.attributes['text']
        if len(self.children):
            txt += '(' + ' '.join(list(map(str, self.children))) + ')'
        return txt

    def __hash__(self, depth=0):
        txt = '(' + str(self.attributes)
        if len(self.children):
            txt += ','
            tab = '\n' + '     '.join([' '] * (depth+1))
            children = [tab + c.__hash__(depth=depth+1) for c in self.children]
            txt += '[' + ','.join(children) + ']'
        return txt + ')'


def spacy_to_tree(node):
    """Get spacy's dependency tree
    e.g.
    nlp = spacy.load('en_core_web_sm')
    doc = nlp('He is here')
    root = next(doc.sents).root
    tree = spacy_to_tree(root)
    """
    attributes = dict(text=node.text, tag=node.pos_,
                      idx=node.i,  dep=node.dep_)
    return Tree(attributes, list(map(spacy_to_tree, node.children)))


def modify_tree(doc, distance=80):
    """Interactive plotting: expect spacy doc structure
    e.g.
    doc = nlp('He is here')
    modify_tree(doc)
    """

    def get_html(doc, init=False):
        html = displacy.render(doc, style="dep", jupyter=False,
                               options=dict(distance=distance))
        # add preview of compressed tree
        html += '\n\n\n' + str(spacy_to_tree(next(doc.sents).root))
        return html

    # Init parse widgets
    html = widgets.HTML(values=get_html(doc, init=True))
    out = widgets.Output()
    edges = [SelectionSlider(options=[w.text + '_%i' % w.i for w in doc],
                             value=str(word.head) + '_%i' % word.head.i,
                             description=str(word))
             for word in doc]
    dep = ('ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos',
           'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound',
           'compound:prt', 'conj', 'cop', 'csubj', 'dative', 'dep', 'det',
           'det:nummod', 'dobj', 'expl', 'expl:pv', 'flat', 'intj', 'iobj',
           'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass',
           'nummod', 'obj', 'obl', 'oprd', 'parataxis', 'pcomp', 'pobj',
           'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod',
           'relcl', 'subtok', 'xcomp')

    dep = [Dropdown(options=dep, value=str(word.dep_), description=' ')
           for word in doc]

    def plot_tree(*args, **kwargs):
        # Update values
        for word, c in zip(doc, edges):
            word.head = [w for w in doc][c.index]

        for word, value in zip(doc, dep):
            word.dep_ = value.value

        html.value = get_html(doc)

    # interaction controls
    controls = np.ravel([z for z in zip(edges, dep)])
    interact(plot_tree, **dict(('w%i' % i, v) for i, v in enumerate(controls)))
    clear_output(wait=True)

    # plot
    display(VBox([HBox([VBox(edges), VBox(dep)]), html]))


class InteractiveTree():
    def __init__(self, sentences, nlp, data_path='./', distance=80):
        self.sentences = sentences
        self.nlp = nlp
        self.data_path = data_path

        self.select = SelectionSlider(options=sentences.index,
                                      description='sentence id')
        self.select.on_trait_change(self.load)

        self.save_button = widgets.Button(description='save')
        self.save_button.on_click(self.save)

        self.reset_button = widgets.Button(description='reset')
        self.reset_button.on_click(self.reset)

        self.load_button = widgets.Button(description='load')
        self.load_button.on_click(self.load)
        self.distance = distance

        self.load()
        self.update()

    def update(self,):
        interact(modify_tree(self.doc_, distance=self.distance),
                 index=self.select,
                 load=self.load_button,
                 reset=self.reset_button)
        display(HBox([self.select, self.save_button, self.load_button,
                      self.reset_button]))

    def reset(self, *args, **kwargs):
        self.doc_ = self.nlp(self.sentences.loc[self.select.value])
        self.update()

    def load(self, *args, **kwargs):
        try:
            fname = os.path.join(self.data_path,
                                 '%i_parsed.json' % self.select.value)
            with open(fname, 'r') as f:
                json = literal_eval(f.readlines()[0])
            doc = self.nlp(json['text'])
            for w, token in zip(doc, json['tokens']):
                w.pos_ = token['pos']
                w.head = doc[token['head']]
                w.dep_ = token['dep']
            print(str(spacy_to_tree(next(doc.sents).root)))
            self.doc_ = doc
        except Exception:
            print('default parsing!')
            self.reset()
        self.update()
        return self.doc_

    def save(self, *args, **kwargs):
        fname = os.path.join(self.data_path,
                             '%i_parsed.json' % self.select.value)
        with open(fname, 'w') as f:
            f.write(str(self.doc_.to_json()))
        print('saving %s' % fname)
        print(str(spacy_to_tree(next(self.doc_.sents).root)))
