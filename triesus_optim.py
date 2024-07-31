###sbatch --time=8:00:00 --cpus-per-task=48 --mem=512G --wrap "python triesus_optim.py"

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import itertools
import xarray as xr
from ortools.sat.python import cp_model
from collections import Counter


# def read_collection(file_tsv: str) -> dict:
#     with open(file_tsv) as file_fh:
#         collection_dict = {}
#         for line_str in file_fh:
#             fields_list = line_str.strip().split("\t")
#             set_id_str = fields_list.pop(0)
#             collection_dict[set_id_str] = fields_list
#     return collection_dict

# class TrieNode:
#     def __init__(self, parent=None, symbol=None):
#         self.parent = parent
#         self.children = {}
#         self.is_end_of_word = False
#         self.words_ending_here = 0
#         self.symbol = symbol

# class Trie:
#     def __init__(self):
#         self.root = TrieNode()
#         self.end_nodes = []
#     #
#     def insert(self, word):
#         current = self.root
#         for letter in word:
#             if letter not in current.children:
#                 current.children[letter] = TrieNode(parent=current, symbol=letter)
#             current = current.children[letter]
#         current.is_end_of_word = True
#         current.words_ending_here = current.words_ending_here + 1
#         self.end_nodes.append(current)
#     #
#     def search(self, word):
#         current = self.root
#         for letter in word:
#             if letter not in current.children:
#                 return False
#             current = current.children[letter]
#         return current.is_end_of_word
#     #
#     def starts_with(self, prefix):
#         current = self.root
#         for letter in prefix:
#             if letter not in current.children:
#                 return False
#             current = current.children[letter]
#         return True
#     #
#     def end_node_for_word(self, word):
#         current = self.root
#         for letter in word:
#             if letter not in current.children:
#                 return None
#             current = current.children[letter]
#         return current
#     #
#     def prefix_from_node(self, node):
#         prefix = []
#         current = node
#         while current.parent != None:
#             prefix.append(current.parent)
#             current = current.parent
#         return prefix


# def solve_cover_set(input_sets):
#     model = cp_model.CpModel()
#     solver = cp_model.CpSolver()
#     # Create a binary variable for each set in the input
#     set_vars = {}
#     for set_name in input_sets:
#         set_vars[set_name] = model.NewBoolVar(set_name)
#     # Create constraints to cover all elements with the selected sets
#     for element in set.union(*input_sets.values()):
#         model.AddBoolOr(
#             [
#                 set_vars[set_name]
#                 for set_name, element_set in input_sets.items()
#                 if element in element_set
#             ]
#         )
#     # Create the objective to minimize the number of selected sets
#     objective = sum(set_vars.values())
#     model.Minimize(objective)
#     # Callback to store all solutions
#     class SolutionCollector(cp_model.CpSolverSolutionCallback):
#         def __init__(self, set_vars):
#             cp_model.CpSolverSolutionCallback.__init__(self)
#             self.set_vars = set_vars
#             self.solutions = []
#         #    
#         def on_solution_callback(self):
#             selected_sets = [
#                 set_name for set_name, var in self.set_vars.items() if self.Value(var) == 1
#             ]
#             self.solutions.append(selected_sets)
#             #print(selected_sets)
#     all_solutions = []
#     min_length = float('inf')
#     found_all_solutions = True
#     while found_all_solutions:
#         # Create the solution collector
#         solution_collector = SolutionCollector(set_vars)
#         # Solve the model with the solution callback
#         status = solver.SolveWithSolutionCallback(model, solution_collector)
#         #print(status)
#         if status == cp_model.OPTIMAL:
#             ### Keep only the shortest solutions
#             for solution in solution_collector.solutions:
#                 solution_length = len(solution)
#                 if solution_length < min_length:
#                     all_solutions = [solution]
#                     min_length = solution_length
#                 elif solution_length == min_length:
#                     all_solutions.append(solution)
#             # Exclude the current solution to find a  new one
#             model.Add(sum(set_vars[set_name] for set_name in solution_collector.solutions[-1]) <= len(solution_collector.solutions[-1]) - 1)
#         else:
#             found_all_solutions = False
#     # Return all collected solutions and the status
#     return all_solutions, status

### default extended 532.2 sec

# def solve_cover_set(input_sets):
#     model = cp_model.CpModel()
#     solver = cp_model.CpSolver()
#     #
#     # Create a binary variable for each set in the input
#     set_vars = {set_name: model.NewBoolVar(set_name) for set_name in input_sets}
#     #
#     # Create constraints to cover all elements with the selected sets
#     all_elements = set.union(*input_sets.values())
#     for element in all_elements:
#         model.AddBoolOr(
#             [set_vars[set_name] for set_name, element_set in input_sets.items() if element in element_set]
#         )
#     #
#     # Create the objective to minimize the number of selected sets
#     model.Minimize(sum(set_vars.values()))
#     #
#     # Callback to store all solutions and filter based on the minimum length
#     class SolutionCollector(cp_model.CpSolverSolutionCallback):
#         def __init__(self, set_vars):
#             cp_model.CpSolverSolutionCallback.__init__(self)
#             self.set_vars = set_vars
#             self.solutions = []
#             self.min_length = float('inf')
#         #
#         def on_solution_callback(self):
#             selected_sets = [set_name for set_name, var in self.set_vars.items() if self.Value(var) == 1]
#             solution_length = len(selected_sets)
#             if solution_length < self.min_length:
#                 self.solutions = [selected_sets]
#                 self.min_length = solution_length
#             elif solution_length == self.min_length:
#                 self.solutions.append(selected_sets)
#     #
#     all_solutions = []
#     min_length = float('inf')
#     found_all_solutions = True
#     #
#     while found_all_solutions:
#         # Create the solution collector
#         solution_collector = SolutionCollector(set_vars)
#         # Solve the model with the solution callback
#         status = solver.SolveWithSolutionCallback(model, solution_collector)
#         #
#         if status == cp_model.OPTIMAL:
#             # Update all_solutions with the current best solutions
#             for solution in solution_collector.solutions:
#                 solution_length = len(solution)
#                 if solution_length < min_length:
#                     all_solutions = [solution]
#                     min_length = solution_length
#                 elif solution_length == min_length:
#                     all_solutions.append(solution)
#             # Exclude the current solution to find a new one
#             model.Add(sum(set_vars[set_name] for set_name in solution_collector.solutions[-1]) <= len(solution_collector.solutions[-1]) - 1)
#         else:
#             found_all_solutions = False
#     #
#     return all_solutions, solver.StatusName(status)

### optimized 1, 551.7 sec

# def solve_cover_set(input_sets):
#     model = cp_model.CpModel()
#     solver = cp_model.CpSolver()
#     #
#     # Create a binary variable for each set in the input
#     set_vars = {set_name: model.NewBoolVar(set_name) for set_name in input_sets}
#     #
#     # Create constraints to cover all elements with the selected sets
#     all_elements = set.union(*input_sets.values())
#     for element in all_elements:
#         model.AddBoolOr([set_vars[set_name] for set_name, element_set in input_sets.items() if element in element_set])
#     #
#     # Create the objective to minimize the number of selected sets
#     objective = sum(set_vars.values())
#     model.Minimize(objective)
#     #
#     class SolutionCollector(cp_model.CpSolverSolutionCallback):
#         def __init__(self, set_vars):
#             super().__init__()
#             self.set_vars = set_vars
#             self.solutions = []
#     #    
#         def on_solution_callback(self):
#             selected_sets = [set_name for set_name, var in self.set_vars.items() if self.Value(var)]
#             self.solutions.append(selected_sets)
#     #
#     all_solutions = []
#     min_length = float('inf')
#     found_all_solutions = True
#     #
#     while found_all_solutions:
#         solution_collector = SolutionCollector(set_vars)
#         status = solver.SolveWithSolutionCallback(model, solution_collector)
#         #    
#         if status == cp_model.OPTIMAL:
#             for solution in solution_collector.solutions:
#                 solution_length = len(solution)
#                 if solution_length < min_length:
#                     all_solutions = [solution]
#                     min_length = solution_length
#                 elif solution_length == min_length:
#                     all_solutions.append(solution)
#             # Add a constraint to exclude the current solution
#             model.Add(sum(set_vars[set_name] for set_name in solution_collector.solutions[-1]) <= len(solution_collector.solutions[-1]) - 1)
#         else:
#             found_all_solutions = False
#     #
#     return all_solutions, status

# ### optimized 2, 531.2 seconds

# class TrieSUS(Trie):
#     def __init__(self, collection: dict):
#         super().__init__()
#         self.item_counts = self.get_item_counts(collection)
#         self.sorted_items = self.sort_keys_by_value(self.item_counts)
#         self.symbol_ranks = self.rank_dict_from_keys_list(self.sorted_items)
#         self.counts_to_symbols = self.reverse_mapping(self.item_counts)
#         self.collection = self.sort_collection_by_other_list_order(
#             collection, self.sorted_items
#         )
#         words = [item for key, item in self.collection.items()]
#         for word in words:
#             self.insert(word)
#     #
#     def reverse_mapping(self, A: dict) -> dict:
#         """Creates a new dictionary B from an existing dictionary A with values as keys and list of corresponding keys as values.
#     #
#         Args:
#             A: A dictionary where different keys are mapped to different values. All keys are different but some values can be the same.
#     #
#         Returns:
#             A new dictionary B where the keys are the values of dictionary A, and the values are the list of corresponding keys in dictionary A.
#         """
#     #
#         B = {}
#     #
#         for key, value in A.items():
#             if value not in B:
#                 B[value] = set()
#     #
#             B[value].add(key)
#     #
#         return B
#     #
#     def transpose_dict(self, A: dict):
#         """
#         Create a new dictionary of sets B, where the keys are the items found in the sets
#         of dictionary A, and the values are the corresponding keys in the dictionary A.
#     #
#         Parameters:
#         - A (dict): The input dictionary of sets.
#     #
#         Returns:
#         - dict: The new dictionary of sets B.
#         """
#         B = {}
#     #
#         for key, value_set in A.items():
#             for item in value_set:
#                 if item not in B:
#                     B[item] = set()
#                 B[item].add(key)
#     #
#         return B
#     #
#     def get_item_counts(self, collection: dict) -> dict:
#         item_list = []
#         for key, item in collection.items():
#             item_list = item_list + item
#         item_count_dict = {}
#         for item_str in item_list:
#             if item_str in item_count_dict:
#                 item_count_dict[item_str] += 1
#             else:
#                 item_count_dict[item_str] = 1
#         return item_count_dict
#     #
#     def find_most_frequent_item(self, list_of_sets):
#         """Returns the item that appears most frequently in a list of sets.
#     #
#         Args:
#           set_list: A list of sets.
#     #
#         Returns:
#           The item that appears most frequently in the sets, or None if the list is empty.
#         """
#     #
#         # Create a counter object to track the frequency of each item.
#         counter = Counter()
#         for set in list_of_sets:
#             for item in set:
#                 counter[item] += 1
#     #
#         most_frequent_item = counter.most_common(1)[0][0]
#     #
#         return most_frequent_item
#     #
#     def sort_keys_by_value(self, dict_obj: dict) -> list:
#         items = list(dict_obj.items())
#         items.sort(key=lambda x: x[1], reverse=True)
#         sorted_keys = [item[0] for item in items]
#         return sorted_keys
#     #
#     def rank_dict_from_keys_list(self, keys_list: list):
#         rank_dict = {}
#         for i, key in enumerate(keys_list):
#             rank_dict[key] = i + 1
#         return rank_dict
#     #
#     def sort_list_by_other_list_order(self, my_list: list, other_list: list) -> list:
#         item_to_index = {
#             item: other_list.index(item) for item in my_list if item in other_list
#         }
#         sorted_list = sorted(
#             my_list, key=lambda x: item_to_index.get(x, len(other_list))
#         )
#         return sorted_list
#     #
#     def sort_collection_by_other_list_order(
#         self, collection_dict: dict, sorted_items_list: list
#     ) -> dict:
#         for key in collection_dict.keys():
#             collection_dict[key] = self.sort_list_by_other_list_order(
#                 collection_dict[key], sorted_items_list
#             )
#         return collection_dict
#     #
#     def get_prefix_symbols(self, node: TrieNode):
#         prefix = [node.symbol]
#         prefix = prefix + [x.symbol for x in self.prefix_from_node(node)]
#         prefix = prefix[:-1]
#         prefix = list(reversed(prefix))
#         return prefix
#     #
#     def get_common_ancestor(self, node_a, node_b):
#         a_nodes = {node_a}
#         node = node_a
#         while node != self.root:
#             node = node.parent
#             a_nodes.add(node)
#     #
#         node = node_b
#         found = False
#         while found == False:
#             if node in a_nodes:
#                 found = True
#             else:
#                 node = node.parent
#     #
#         return node
#     #
#     def find_sus(self, word):
#         word_end_node = self.end_node_for_word(word)
#         other_end_nodes = [node for node in self.end_nodes if node != word_end_node]
#     #
#         if len(word_end_node.children) != 0 or word_end_node.words_ending_here > 1:
#             # if word_end_node has children, or there is an identical word,
#             # the SUS doesn't exist, terminate here
#             return []
#     #
#         candidate_symbols = []  # list of sets
#     #
#         for end_node in other_end_nodes:
#             current_word_node = word_end_node
#     #
#             unique_items = set()
#     #
#             trie_word = set(self.get_prefix_symbols(end_node))
#     #
#             # determine where the two words intersect in the trie
#             common_ancestor_node = self.get_common_ancestor(word_end_node, end_node)
#     #
#             while current_word_node != common_ancestor_node:
#                 if current_word_node.symbol not in trie_word:
#                     unique_items.add(current_word_node.symbol)
#     #
#                 current_word_node = current_word_node.parent
#     #
#             if len(unique_items) == 0:
#                 return []
#     #
#             candidate_symbols.append(unique_items)
#     #
#         candidates_dict = {}
#         for i in range(len(candidate_symbols)):
#             candidates_dict[i] = candidate_symbols[i]
#         sets_to_cover = self.transpose_dict(candidates_dict)
#     #
#         sus, status = solve_cover_set(sets_to_cover)
#     #
#         return sus

# def run_triesus(collection_tsv: str) -> None:
#     collection_dict = read_collection(collection_tsv)
#     tic = time.perf_counter()
#     triesus = TrieSUS(collection_dict)
#     for key, item in triesus.collection.items():
#         #print(item)
#         sus = triesus.find_sus(item)
#         sus = ' '.join([str(elem) for i,elem in enumerate(sus)])
#         print(f"{key}\t{sus}")
#     toc = time.perf_counter()
#     print(f"Elapsed time: {toc - tic:0.3f} seconds")


# example1 = "random_sets/10_10_10.tsv"
# run_triesus(example1) 
# example2 = "random_sets/20_20_20.tsv"
# run_triesus(example2) 
# example3 = "random_sets/30_30_30.tsv"
# run_triesus(example3) 
# example4 = "random_sets/40_40_40.tsv"
# run_triesus(example4)
# example5 = "random_sets/50_50_50.tsv"
# run_naive_sus(example5)
# example6 = "random_sets/60_60_60.tsv"
# run_naive_sus(example6)
# example7 = "random_sets/70_70_70.tsv"
# run_naive_sus(example7)v

class TrieNode:
    def __init__(self, parent=None, symbol=None):
        self.parent = parent
        self.children = {}
        self.is_end_of_word = False
        self.words_ending_here = 0
        self.symbol = symbol

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.end_nodes = []
    #
    def insert(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode(parent=current, symbol=letter)
            current = current.children[letter]
        current.is_end_of_word = True
        current.words_ending_here += 1
        self.end_nodes.append(current)
    #
    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                return False
            current = current.children[letter]
        return current.is_end_of_word
    #
    def starts_with(self, prefix):
        current = self.root
        for letter in prefix:
            if letter not in current.children:
                return False
            current = current.children[letter]
        return True
    #
    def end_node_for_word(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                return None
            current = current.children[letter]
        return current
    #
    def prefix_from_node(self, node):
        prefix = []
        current = node
        while current.parent is not None:
            prefix.append(current.symbol)
            current = current.parent
        return prefix

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
        #   
        # Insert words into the Trie in parallel
        self.insert_words_parallel(words)
    #
    def insert_words_parallel(self, words):
        with ThreadPoolExecutor() as executor:
            list(executor.map(self.insert, words))
    #
    def reverse_mapping(self, A: dict) -> dict:
        B = {}
        for key, value in A.items():
            if value not in B:
                B[value] = set()
            B[value].add(key)
        return B
    #
    def transpose_dict(self, A: dict):
        B = {}
        for key, value_set in A.items():
            for item in value_set:
                if item not in B:
                    B[item] = set()
                B[item].add(key)
        return B
    #
    def get_item_counts(self, collection: dict) -> dict:
        item_list = []
        for key, item in collection.items():
            item_list += item
        return Counter(item_list)
    #
    def sort_keys_by_value(self, dict_obj: dict) -> list:
        return sorted(dict_obj, key=dict_obj.get, reverse=True)
    #
    def rank_dict_from_keys_list(self, keys_list: list):
        return {key: i + 1 for i, key in enumerate(keys_list)}
    #
    def sort_list_by_other_list_order(self, my_list: list, other_list: list) -> list:
        item_to_index = {item: other_list.index(item) for item in my_list if item in other_list}
        return sorted(my_list, key=lambda x: item_to_index.get(x, len(other_list)))
    #
    def sort_collection_by_other_list_order(self, collection_dict: dict, sorted_items_list: list) -> dict:
        for key in collection_dict.keys():
            collection_dict[key] = self.sort_list_by_other_list_order(collection_dict[key], sorted_items_list)
        return collection_dict
    #
    def get_prefix_symbols(self, node: TrieNode):
        return list(reversed([node.symbol] + self.prefix_from_node(node)))
    #
    def get_common_ancestor(self, node_a, node_b):
        a_nodes = {node_a}
        while node_a != self.root:
            node_a = node_a.parent
            a_nodes.add(node_a)
        while node_b not in a_nodes:
            node_b = node_b.parent
        return node_b
    #
    def find_sus(self, word):
        word_end_node = self.end_node_for_word(word)
        other_end_nodes = [node for node in self.end_nodes if node != word_end_node]
    #
        if word_end_node is None or word_end_node.children or word_end_node.words_ending_here > 1:
            return []
    #
        candidate_symbols = []
        for end_node in other_end_nodes:
            unique_items = set()
            trie_word = set(self.get_prefix_symbols(end_node))
            common_ancestor_node = self.get_common_ancestor(word_end_node, end_node)
            current_word_node = word_end_node
            while current_word_node != common_ancestor_node:
                if current_word_node.symbol not in trie_word:
                    unique_items.add(current_word_node.symbol)
                current_word_node = current_word_node.parent
            if not unique_items:
                return []
            candidate_symbols.append(unique_items)
    #
        candidates_dict = {i: symbols for i, symbols in enumerate(candidate_symbols)}
        sets_to_cover = self.transpose_dict(candidates_dict)
        sus, status = solve_cover_set(sets_to_cover)
        return sus

def solve_cover_set(input_sets):
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    #
    set_vars = {set_name: model.NewBoolVar(set_name) for set_name in input_sets}
    all_elements = set.union(*input_sets.values())
    for element in all_elements:
        model.AddBoolOr([set_vars[set_name] for set_name, element_set in input_sets.items() if element in element_set])
    #
    objective = sum(set_vars.values())
    model.Minimize(objective)
    #
    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, set_vars):
            super().__init__()
            self.set_vars = set_vars
            self.solutions = []
    #
        def on_solution_callback(self):
            self.solutions.append([set_name for set_name, var in self.set_vars.items() if self.Value(var)])
    #
    all_solutions = []
    min_length = float('inf')
    found_all_solutions = True
    #
    while found_all_solutions:
        solution_collector = SolutionCollector(set_vars)
        status = solver.SolveWithSolutionCallback(model, solution_collector)
        if status == cp_model.OPTIMAL:
            for solution in solution_collector.solutions:
                solution_length = len(solution)
                if solution_length < min_length:
                    all_solutions = [solution]
                    min_length = solution_length
                elif solution_length == min_length:
                    all_solutions.append(solution)
            model.Add(sum(set_vars[set_name] for set_name in solution_collector.solutions[-1]) <= len(solution_collector.solutions[-1]) - 1)
        else:
            found_all_solutions = False
    #
    return all_solutions, status

def read_collection(file_path):
    with open(file_path, 'r') as file:
        return {line.split()[0]: line.split()[1:] for line in file}

def process_item(args):
    key, item, triesus = args
    sus = triesus.find_sus(item)
    sus_str = ' '.join([str(elem) for elem in sus])
    return f"{key}\t{sus_str}"

def run_triesus(collection_tsv: str) -> None:
    collection_dict = read_collection(collection_tsv)
    tic = time.perf_counter()
    triesus = TrieSUS(collection_dict)
    #
    with Pool(cpu_count()) as pool:
        results = pool.map(process_item, [(key, item, triesus) for key, item in triesus.collection.items()])
    #
    for result in results:
        print(result)
    #
    toc = time.perf_counter()
    print(f"Elapsed time: {toc - tic:0.3f} seconds")

### 157 seconds with multithreading

example1 = "random_sets/10_10_10.tsv"
run_triesus(example1) 
example2 = "random_sets/20_20_20.tsv"
run_triesus(example2) 
example3 = "random_sets/30_30_30.tsv"
run_triesus(example3) 
example4 = "random_sets/40_40_40.tsv"
run_triesus(example4)