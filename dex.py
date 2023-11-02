import pandas as pd
import numpy as np

import treelib
import itertools

class DEX:
    '''
    DEX Model for qualitative decision making
    
    One can create by hand DEX decision model by developing hierarchy and adding decision rules. 
    Decision Rules are propagated from leaf of the hierarchy to root calculating final decision.
    Therefore, DEX is used for classification.
    
    TODO: 
    - Add order of attributes (increasing, decreasing, unordered)
    - Checking for monotonocity
    - Data-driven hierarchy creation

    '''
    
    def __init__(self, X = None, y = None):
        '''
        Parameters
        ----------
        X : pandas.DataFrame
            Data used for DEX model
        '''
        
        if X is not None:
            self.X = X
        
            for j in self.X.columns:
                self._levels.append([j, self.data.loc[:, j].unique().tolist()])
        if y is not None:
            self.y = y
        
        self.hierarchy = pd.DataFrame(columns=['child', 'parent'])
    
        self._levels = list()
        self._orders = list()
        self._decision_tables = list()
        self._tree_hierarchy = treelib.Tree()
        self._label = ''
    
    # HIERARCHY CREATOR PART
    def add_node(self, children, parent, levels, order='increasing'):
        '''
        Create a hierarchy using a children-parent notation
        
        Adds a set of nodes corresponding to one hierarchy level
        
        Parameters
        ----------
        children : numpy.list
            Set of attribute names that correspond to children attributes (must exists in the data)
        parent : numpy.list
            Set of attribute names (however, it should be only one) that correspond to parent attribute (must NOT exists in data)
        levels : numpy.list
            Ordered levels of the parent attribute
        order : string
            Order of levels. Possible values {increasing, decreasing}
        '''
        
        self._levels.append([parent, levels])
        self._orders.append([parent, order])
        
        nodes = pd.DataFrame([element for element in itertools.product(children, [parent])], columns=['child', 'parent'])
        self.hierarchy = pd.concat([self.hierarchy, nodes])
        
        self.hierarchy = self.hierarchy.reset_index(drop=True)
        
        return
    
    def conclude_hierarchy(self, label, levels, order='increasing'):
        '''
        Add final node to hierarchy connecting the roots to the label attribute
        
        Parameters
        ----------
        label : string
            Label attribute to which other should be connected
        levels: numpy.list
            Ordered levels of the parent attribute
        order : string
            Order of levels. Possible values {increasing, decreasing}
        '''
        
        # FILL KNOWN HIERARCHY
        roots = np.setdiff1d(self.hierarchy['parent'], self.hierarchy['child'])
        
        if roots.shape[0] != 0:
            nodes = pd.DataFrame([element for element in itertools.product(roots, [label])], columns=['child', 'parent'])
            
            self.hierarchy = pd.concat([self.hierarchy, nodes])
            self.hierarchy = self.hierarchy.reset_index(drop=True)
        
        # IF ATTRIBUTE IS NOT IN THE HIERARCHY LINK TO LABEL
        for name, level in self._levels:
            if (name != self._label):
                all_elements = [*np.unique(self.hierarchy['parent']), *np.unique(self.hierarchy['child'])]
                if not (name in all_elements):
                    nodes = pd.DataFrame(columns=['child', 'parent'])
                    nodes.loc[0] = [name, label]
                    self.hierarchy = pd.concat([self.hierarchy, nodes])
                    self.hierarchy = self.hierarchy.reset_index(drop=True)
        
        self._levels.append([label, levels])
        self._orders.append([label, order])
        self._label = label
        
        # TREELIB NOTATION
        roots = np.setdiff1d(self.hierarchy['parent'], self.hierarchy['child'])
        
        self._tree_hierarchy.create_node(roots[0], roots[0], data={'levels': levels, 'order': order})
        self._return_subtree(roots[0])
        
        return
    
    def _return_subtree(self, attribute):
        '''
        Hidden recursive method for tree creation
        
        Parameters
        ----------
        attribute : string
            Name of the attribute for which tree needs to be extracted
        '''
        subset = self.hierarchy.loc[self.hierarchy['parent'] == attribute, ]
        for _, row in subset.iterrows():
            for i, j in self._levels:
                if row['child'] == i:
                    levels_to_add = j
            for i, j in self._orders:
                if row['child'] == i:
                    order_to_add = j
            
            self._tree_hierarchy.create_node(row['child'], row['child'], parent=row['parent'], data={'levels': levels_to_add, 'order': order_to_add})
            self._return_subtree(row['child'])
        return
    
    def show_hierarchy(self):
        '''
        Show hierarchy in Treelib notation
        
        '''
        
        self._tree_hierarchy.show()
        
        return
    
    # DECISION RULES CREATION
    def create_dataset_for_decision_rules(self, attribute):
        '''
        Create dataset for decision rules for parent attribute
        
        Parameters
        ----------
        attribute: string
            Attribute for which dataset needs to be created
        '''
        
        values_for_array = list()
        names_of_columns = []
        
        attributes = self.hierarchy.loc[self.hierarchy['parent'] == attribute, 'child'].tolist()
        
        for att in attributes:
            for i, j in self._levels:
                if att == i:
                    names_of_columns.append(att)
                    values_for_array.append(j)
        
        df = pd.DataFrame([element for element in itertools.product(*values_for_array)], columns=names_of_columns)
        df[attribute] = ''
        return df
    
    def add_decision_table(self, attribute, df):
        '''
        Add DEX decision table to model
        
        Parameters
        ----------
        attribute: string
            Attribute for which DEX decision table needs to be added
        df: pandas.DataFrame
            DEX decision table
        '''
        
        self._decision_tables.append([attribute, df])
        self._tree_hierarchy.get_node(attribute).data['decision_table'] = df
        
        return
    
    #FIT
    def fit(self, X = None, y = None):
        '''
        Fit data to model for probability calculation
        
        Parameters
        ----------
        X : pandas.DataFrame
            Dataset used for model augmentation
        y : pandas.Series
            Labeled outcome
        '''
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y
        
        decision_table = self._tree_hierarchy.get_node(self._label).data['decision_table']
       
        model = self.predict(X, return_intermediate=True)
        model = model[decision_table.columns.tolist()]
        model[self._label] = y.values
        model = pd.get_dummies(data=model, columns=[decision_table.columns[-1]], prefix='', prefix_sep='')
        model = model.groupby(decision_table.columns[:-1].tolist()).agg('sum')
        
        self.model = model
        
        return self
    
    #PREDICT
    def predict(self, df, return_intermediate=False):
        '''
        Apply DEX decision rules and provide output 
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataset to which predict should be applied
        return_intermediate : bool
            Signal whether intermediate features should be calculated
        '''
        
        applied_hierarchy = self._tree_hierarchy._clone(with_tree=True, deep=True)
        while applied_hierarchy.size() > 1:
            leaves = [l.identifier for l in applied_hierarchy.leaves()]
            
            parents = [applied_hierarchy.parent(l).identifier for l in leaves]
            parents = list(dict.fromkeys(parents))

            for p in parents:
                decision_table = applied_hierarchy.get_node(p).data['decision_table']
                
                children = [c.identifier for c in applied_hierarchy.children(p)]
#                 df = df.merge(decision_table, on=children, suffixes=('', '_predicted'), how='left')
                
#                 # REMOVE NODE FROM HIERARCHY
#                 for c in children:
#                     applied_hierarchy.remove_node(c)
                    

                if set(children).issubset(set(df.columns)):
                    df = df.merge(decision_table, on=children, suffixes=('', '_predicted'), how='left')
                    
                    # REMOVE NODE FROM HIERARCHY
                    for c in children:
                        applied_hierarchy.remove_node(c)
        
        if return_intermediate:
            return df
        else:
            return df.iloc[:, -1]
        
    def predict_proba(self, df, smoothing = 0):
        '''
        Assign probability of of event for each class based on given data X and y
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataset for which probabilities needs to be calculated
        smoothing : int
            Parameter for probability smoothing
        '''
        
        decision_table = self._tree_hierarchy.get_node(self._label).data['decision_table']
        label_levels = self._tree_hierarchy.get_node(self._label).data['levels']
        self.model[label_levels.astype(str)] = self.model[label_levels.astype(str)].apply(lambda x: (x + smoothing), axis=1)
        self.model[label_levels.astype(str)] = self.model[label_levels.astype(str)].apply(lambda x: x/x.sum(), axis=1)
        self.model = self.model.reset_index()
        
        predicted = self.predict(df, return_intermediate=True)
        predicted = predicted.merge(self.model, on = decision_table.columns[:-1].tolist())
        
#         return predicted.loc[:, label_levels]
        return predicted.iloc[:, -len(label_levels):]
        
    # LEARN FROM DATA
    def add_data(self, X, y):
        '''
        Add data to current model
        
        Parameters
        ----------
        X : pandas.DataFrame
            Pandas DataFrame that corresponds to entered DEX model
        y : pandas.Series
            Output of the model
        '''
        
        self.X = X
        self.y = y
        return
    
    def aic(self):
        '''
        Calculate Akaike information criteria
        '''
        
        complexity = 0
        for n in self._tree_hierarchy.all_nodes():
            if 'decision_table' in n.data.keys():
                #complexity = complexity + np.ceil(np.log2(n.data['decision_table'].shape[0]))
                complexity = complexity + n.data['decision_table'].shape[0]
        
        label_decision_table = self._tree_hierarchy.get_node(self._label).data['decision_table']
        
        pred = self.predict_proba(self.X, smoothing=0)
#         pred = pred.apply(lambda x: np.log(x) if x > 0 else 0)
        pred = np.log(pred.mask(pred <= 0)).fillna(0)
        
        target = pd.get_dummies(self.y)
        target = target[pred.columns]
        
        likelihood = -1 * (target * pred).sum().sum()/target.shape[0]
        
        return list([2 * (complexity - likelihood), complexity, likelihood])
    
    def bic(self):
        '''
        Calculate Bayesian information criteria
        '''
        
        complexity = 0
        for n in self._tree_hierarchy.all_nodes():
            if 'decision_table' in n.data.keys():
                complexity = complexity + n.data['decision_table'].shape[0]
        
        #label_decision_table = self._tree_hierarchy.get_node(self._label).data['decision_table']
        
        pred = self.predict_proba(self.X, smoothing=0)
        pred = np.log2(pred)
        pred[np.isneginf(pred)] = 0
        
        target = pd.get_dummies(self.y)

        vals = np.multiply(target, pred)
        
        likelihood = -1 * (vals).sum().sum()/target.shape[0]
        
        return list([complexity * np.log(target.shape[0]) - 2 * likelihood, complexity, likelihood])