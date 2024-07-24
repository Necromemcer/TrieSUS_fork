
### init.py
def read_collection(file_tsv: str) -> dict:
    with open(file_tsv) as file_fh:
        collection_dict = {}
        for line_str in file_fh:
            fields_list = line_str.strip().split("\t")
            set_id_str = fields_list.pop(0)
            collection_dict[set_id_str] = fields_list
    return collection_dict


def read_incidence_matrix(file_tsv: str) -> dict:
    with open(file_tsv) as file_fh:
        collection_dict = {}
        for line_str in file_fh:
            fields_list = line_str.strip().split("\t")
            set_id_str = fields_list.pop(0)
            collection_dict[set_id_str] = []
            for i in range(len(fields_list)):
                if int(fields_list[i]) == 1:
                    collection_dict[set_id_str].append(str(i))
    return collection_dict

example = "/nfs/research/petsalaki/users/platon/SPMap/TrieSUS_fork/tests/examples/sets5.tsv"

### cover_set_extended.py

from ortools.sat.python import cp_model

def solve_cover_set(input_sets):
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    # Create a binary variable for each set in the input
    set_vars = {}
    for set_name in input_sets:
        set_vars[set_name] = model.NewBoolVar(set_name)
    # Create constraints to cover all elements with the selected sets
    for element in set.union(*input_sets.values()):
        model.AddBoolOr(
            [
                set_vars[set_name]
                for set_name, element_set in input_sets.items()
                if element in element_set
            ]
        )
    # Create the objective to minimize the number of selected sets
    objective = sum(set_vars.values())
    model.Minimize(objective)
    # Callback to store all solutions
    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, set_vars):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.set_vars = set_vars
            self.solutions = []
        
        def on_solution_callback(self):
            selected_sets = [
                set_name for set_name, var in self.set_vars.items() if self.Value(var) == 1
            ]
            self.solutions.append(selected_sets)
            #print(selected_sets)
    all_solutions = []
    min_length = float('inf')
    found_all_solutions = True
    while found_all_solutions:
        # Create the solution collector
        solution_collector = SolutionCollector(set_vars)
        # Solve the model with the solution callback
        status = solver.SolveWithSolutionCallback(model, solution_collector)
        #print(status)
        if status == cp_model.OPTIMAL:
            for solution in solution_collector.solutions:
                solution_length = len(solution)
                if solution_length < min_length:
                    all_solutions = [solution]
                    min_length = solution_length
                elif solution_length == min_length:
                    all_solutions.append(solution)
            # Exclude the current solution to find a  new one
            model.Add(sum(set_vars[set_name] for set_name in solution_collector.solutions[-1]) <= len(solution_collector.solutions[-1]) - 1)
        else:
            found_all_solutions = False
    # Return all collected solutions and the status
    return all_solutions, status

### trie.py

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.end_nodes = []
    ###
    def insert(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode(parent=current, symbol=letter)
            current = current.children[letter]
        current.is_end_of_word = True
        current.words_ending_here = current.words_ending_here + 1
        self.end_nodes.append(current)
    ###
    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                return False
            current = current.children[letter]
        return current.is_end_of_word
    ###
    def starts_with(self, prefix):
        current = self.root
        for letter in prefix:
            if letter not in current.children:
                return False
            current = current.children[letter]
        return True
    ###
    def end_node_for_word(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                return None
            current = current.children[letter]
        return current
    ###
    def prefix_from_node(self, node):
        prefix = []
        current = node
        while current.parent != None:
            prefix.append(current.parent)
            current = current.parent
        return prefix

### trienode.py

class TrieNode:
    def __init__(self, parent=None, symbol=None):
        self.parent = parent
        self.children = {}
        self.is_end_of_word = False
        self.words_ending_here = 0
        self.symbol = symbol

### triesus.py

from collections import Counter

class TrieSUS(Trie):
    def __init__(self, collection: dict):
        super().__init__()
        self.item_counts = self.get_item_counts(collection)
        self.sorted_items = self.sort_keys_by_value(self.item_counts)
        self.symbol_ranks = self.rank_dict_from_keys_list(self.sorted_items)
        self.counts_to_symbols = self.reverse_mapping(self.item_counts)
        self.collection = self.sort_collection_by_other_list_order(
            collection, self.sorted_items
        )
        words = [item for key, item in self.collection.items()]
        for word in words:
            self.insert(word)
    ###
    def reverse_mapping(self, A: dict) -> dict:
        """Creates a new dictionary B from an existing dictionary A with values as keys and list of corresponding keys as values.
    ###
        Args:
            A: A dictionary where different keys are mapped to different values. All keys are different but some values can be the same.
    ###
        Returns:
            A new dictionary B where the keys are the values of dictionary A, and the values are the list of corresponding keys in dictionary A.
        """
    ###
        B = {}
    ###
        for key, value in A.items():
            if value not in B:
                B[value] = set()
    ###
            B[value].add(key)
    ###
        return B
    ###
    def transpose_dict(self, A: dict):
        """
        Create a new dictionary of sets B, where the keys are the items found in the sets
        of dictionary A, and the values are the corresponding keys in the dictionary A.
    ###
        Parameters:
        - A (dict): The input dictionary of sets.
    ###
        Returns:
        - dict: The new dictionary of sets B.
        """
        B = {}
    ###
        for key, value_set in A.items():
            for item in value_set:
                if item not in B:
                    B[item] = set()
                B[item].add(key)
    ###
        return B
    ###
    def get_item_counts(self, collection: dict) -> dict:
        item_list = []
        for key, item in collection.items():
            item_list = item_list + item
        item_count_dict = {}
        for item_str in item_list:
            if item_str in item_count_dict:
                item_count_dict[item_str] += 1
            else:
                item_count_dict[item_str] = 1
        return item_count_dict
    ###
    def find_most_frequent_item(self, list_of_sets):
        """Returns the item that appears most frequently in a list of sets.
    ###
        Args:
          set_list: A list of sets.
    ###
        Returns:
          The item that appears most frequently in the sets, or None if the list is empty.
        """
    ###
        # Create a counter object to track the frequency of each item.
        counter = Counter()
        for set in list_of_sets:
            for item in set:
                counter[item] += 1
    ###
        most_frequent_item = counter.most_common(1)[0][0]
    ###
        return most_frequent_item
    ###
    def sort_keys_by_value(self, dict_obj: dict) -> list:
        items = list(dict_obj.items())
        items.sort(key=lambda x: x[1], reverse=True)
        sorted_keys = [item[0] for item in items]
        return sorted_keys
    ###
    def rank_dict_from_keys_list(self, keys_list: list):
        rank_dict = {}
        for i, key in enumerate(keys_list):
            rank_dict[key] = i + 1
        return rank_dict
    ###
    def sort_list_by_other_list_order(self, my_list: list, other_list: list) -> list:
        item_to_index = {
            item: other_list.index(item) for item in my_list if item in other_list
        }
        sorted_list = sorted(
            my_list, key=lambda x: item_to_index.get(x, len(other_list))
        )
        return sorted_list
    ###
    def sort_collection_by_other_list_order(
        self, collection_dict: dict, sorted_items_list: list
    ) -> dict:
        for key in collection_dict.keys():
            collection_dict[key] = self.sort_list_by_other_list_order(
                collection_dict[key], sorted_items_list
            )
        return collection_dict
    ###
    def get_prefix_symbols(self, node: TrieNode):
        prefix = [node.symbol]
        prefix = prefix + [x.symbol for x in self.prefix_from_node(node)]
        prefix = prefix[:-1]
        prefix = list(reversed(prefix))
        return prefix
    ###
    def get_common_ancestor(self, node_a, node_b):
        a_nodes = {node_a}
        node = node_a
        while node != self.root:
            node = node.parent
            a_nodes.add(node)
    ###
        node = node_b
        found = False
        while found == False:
            if node in a_nodes:
                found = True
            else:
                node = node.parent
    ###
        return node
    ###
    def find_sus(self, word):
        word_end_node = self.end_node_for_word(word)
        other_end_nodes = [node for node in self.end_nodes if node != word_end_node]
    ###
        if len(word_end_node.children) != 0 or word_end_node.words_ending_here > 1:
            # if word_end_node has children, or there is an identical word,
            # the SUS doesn't exist, terminate here
            return []
    ###
        candidate_symbols = []  # list of sets
    ###
        for end_node in other_end_nodes:
            current_word_node = word_end_node
            unique_items = set()
            trie_word = set(self.get_prefix_symbols(end_node))
            # determine where the two words intersect in the trie
            common_ancestor_node = self.get_common_ancestor(word_end_node, end_node)
    ###
            while current_word_node != common_ancestor_node:
                if current_word_node.symbol not in trie_word:
                    unique_items.add(current_word_node.symbol)
    ###
                current_word_node = current_word_node.parent
    ###
            if len(unique_items) == 0:
                return []
    ###
            candidate_symbols.append(unique_items)
    ###
        candidates_dict = {}
        for i in range(len(candidate_symbols)):
            candidates_dict[i] = candidate_symbols[i]
        sets_to_cover = self.transpose_dict(candidates_dict)
        #print(sets_to_cover)
    ###
        sus, status = solve_cover_set(sets_to_cover)
        #print(sus)
    ###
        return sus

### run_triesus.py

def run_triesus_tsv(collection_tsv: str) -> None:
    collection_dict = read_collection(collection_tsv)
    triesus = TrieSUS(collection_dict)
    for key, item in triesus.collection.items():
        #print(item)
        sus = triesus.find_sus(item)
        sus = ' '.join([str(elem) for i,elem in enumerate(sus)])
        print(f"{key}\t{sus}")

def run_triesus_pipeline(collection_dict: str) -> dict:
    #collection_dict = read_collection(collection_tsv)
    triesus = TrieSUS(collection_dict)
    result = {}
    for key, item in triesus.collection.items():
        sus = triesus.find_sus(item)
        #sus = "\t".join(sus)
        #print(f"{key}\t{sus}")
        result[key] = sus
    return result

### naive_sus_extended.py

def naive_sus_extended(sets_dict: dict, set_key: str):
    s = sets_dict[set_key]
    other_sets_keys = list(sets_dict.keys())
    other_sets_keys.remove(set_key)
    equisolutions = []
    for n in range(1, len(s) + 1):
        # Initialize a binary mask with the first n elements set to 1
        mask = (1 << n) - 1
        while mask < (1 << len(s)):
            subset = []
            # Iterate through the elements of the set and check if the corresponding
            # bit in the mask is set (1), if yes, add the element to the subset
            for i in range(len(s)):
                if (mask >> i) & 1:
                    subset.append(s[i])
            #print(subset)
            # check whether subset is sus
            sets_with_such_elements_number = 0
            for other_set_key in other_sets_keys:
                other_set = sets_dict[other_set_key]
                if all(e in other_set for e in subset):
                    sets_with_such_elements_number += 1
            if sets_with_such_elements_number == 0:
                equisolutions.append(subset)
                if len(subset) > min([len(i) for i in equisolutions]):
                    equisolutions.remove(subset)
            # Generate the next binary number with n 1s
            # This algorithm is known as Gosper's Hack
            c = mask & -mask
            r = mask + c
            mask = (((r ^ mask) >> 2) // c) | r
    return equisolutions


def naive_sus(sets_dict: dict):
    for set_key in sets_dict.keys():
        sus = naive_sus_extended(sets_dict, set_key)
        sus = ' '.join([str(elem) for i,elem in enumerate(sus)])
        print(f"{set_key}\t{sus}")

sc_example = "/nfs/research/petsalaki/users/platon/SPMap/TrieSUS_fork/tests/examples/sets_sc.tsv"