###sbatch --time=8:00:00 --cpus-per-task=48 --mem=64G --wrap "python triesus_optim.py"

import time
#import concurrent.futures
from multiprocessing import Pool, cpu_count
import itertools

def read_collection(file_tsv: str) -> dict:
    with open(file_tsv) as file_fh:
        collection_dict = {}
        for line_str in file_fh:
            fields_list = line_str.strip().split("\t")
            set_id_str = fields_list.pop(0)
            collection_dict[set_id_str] = fields_list
    return collection_dict

    
# def find_sus(sets_dict: dict, set_key: str):
#     s = sets_dict[set_key]
#     other_sets_keys = list(sets_dict.keys())
#     other_sets_keys.remove(set_key)
#     equisolutions = []
#     for n in range(1, len(s) + 1):
#         # Initialize a binary mask with the first n elements set to 1
#         mask = (1 << n) - 1
#         while mask < (1 << len(s)):
#             subset = []
#             # Iterate through the elements of the set and check if the corresponding
#             # bit in the mask is set (1), if yes, add the element to the subset
#             for i in range(len(s)):
#                 if (mask >> i) & 1:
#                     subset.append(s[i])
#             #print(subset)
#             # check whether subset is sus
#             sets_with_such_elements_number = 0
#             for other_set_key in other_sets_keys:
#                 other_set = sets_dict[other_set_key]
#                 if all(e in other_set for e in subset):
#                     sets_with_such_elements_number += 1
#             if sets_with_such_elements_number == 0:
#                 equisolutions.append(subset)
#                 if len(subset) > min([len(i) for i in equisolutions]):
#                     equisolutions.remove(subset)
#             # Generate the next binary number with n 1s
#             # This algorithm is known as Gosper's Hack
#             c = mask & -mask
#             r = mask + c
#             mask = (((r ^ mask) >> 2) // c) | r
#     return equisolutions

### 709 seconds

# def find_sus(sets_dict: dict, set_key: str):
#     s = sets_dict[set_key]
#     s_len = len(s)
#     other_sets_keys = list(sets_dict.keys())
#     other_sets_keys.remove(set_key)
#     other_sets = [set(sets_dict[key]) for key in other_sets_keys]
#     equisolutions = []
#     min_length = float('inf')
#     for n in range(1, s_len + 1):
#         mask = (1 << n) - 1
#         while mask < (1 << s_len):
#             subset = [s[i] for i in range(s_len) if (mask >> i) & 1]
#             # Check if the subset is not a subset of any other set
#             if all(not set(subset).issubset(other_set) for other_set in other_sets):
#                 subset_length = len(subset)
#                 if subset_length < min_length:
#                     equisolutions = [subset]
#                     min_length = subset_length
#                 elif subset_length == min_length:
#                     equisolutions.append(subset)
#             # Generate the next binary number with n 1s using Gosper's Hack
#             c = mask & -mask
#             r = mask + c
#             mask = (((r ^ mask) >> 2) // c) | r
#     return equisolutions

# def naive_sus(sets_dict: dict):
#     for set_key in sets_dict.keys():
#         sus = find_sus(sets_dict, set_key)
#         for i in sus: 
#         	print(f"{set_key}\t{str(i)}")

### 289 seconds o_0 GPT cool (optimized find_sus)

# def generate_subsets(s):
#     subsets = []
#     for i in range(1, len(s) + 1):
#         for subset in itertools.combinations(s, i):
#             subsets.append(subset)
#     return subsets

# def find_sus(sets_dict: dict, set_key: str):
#     s = sets_dict[set_key]
#     other_sets_keys = list(sets_dict.keys())
#     other_sets_keys.remove(set_key)
#     other_sets = [set(sets_dict[key]) for key in other_sets_keys]
#     equisolutions = []
#     min_length = float('inf')
#     ###
#     subsets = generate_subsets(s)
#     ###
#     for subset in subsets:
#         # Check if the subset is not a subset of any other set
#         if all(not set(subset).issubset(other_set) for other_set in other_sets):
#             subset_length = len(subset)
#             if subset_length < min_length:
#                 equisolutions = [subset]
#                 min_length = subset_length
#             elif subset_length == min_length:
#                 equisolutions.append(subset)
#     return equisolutions

### 241 secons with updated find_sus + generate_subsets

# def naive_sus(sets_dict: dict):
#     results = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_to_key = {executor.submit(find_sus, sets_dict, set_key): set_key for set_key in sets_dict.keys()}
#         for future in concurrent.futures.as_completed(future_to_key):
#             set_key = future_to_key[future]
#             try:
#                 sus = future.result()
#                 results.append((set_key, sus))
#             except Exception as exc:
#                 print(f'{set_key} generated an exception: {exc}')
#     return results

### 284 seconds with multithreading 

# def run_naive_sus(collection_tsv: str):
# 	tic = time.perf_counter()
# 	collection_dict = read_collection(collection_tsv)
# 	naive_sus(collection_dict)
# 	toc = time.perf_counter()
# 	print(f"Elapsed time:{toc-tic:0.3f} seconds")

def run_naive_sus(collection_tsv: str):
    tic = time.perf_counter()
    collection_dict = read_collection(collection_tsv)
    results = naive_sus(collection_dict)
    for set_key, sus_list in results:
        for sus in sus_list:
            print(f"{set_key}\t{str(sus)}")
    toc = time.perf_counter()
    print(f"Elapsed time: {toc - tic:0.3f} seconds")

def generate_subsets(s):
    subsets = []
    for i in range(1, len(s) + 1):
        for subset in itertools.combinations(s, i):
            subsets.append(subset)
    return subsets

def find_sus_worker(args):
    sets_dict, set_key = args
    s = sets_dict[set_key]
    other_sets_keys = list(sets_dict.keys())
    other_sets_keys.remove(set_key)
    other_sets = [set(sets_dict[key]) for key in other_sets_keys]
    equisolutions = []
    min_length = float('inf')
    subsets = generate_subsets(s)
    for subset in subsets:
        # Check if the subset is not a subset of any other set
        if all(not set(subset).issubset(other_set) for other_set in other_sets):
            subset_length = len(subset)
            if subset_length < min_length:
                equisolutions = [subset]
                min_length = subset_length
            elif subset_length == min_length:
                equisolutions.append(subset)
    return set_key, equisolutions

def naive_sus(sets_dict: dict):
    with Pool(cpu_count()) as pool:
        results = pool.map(find_sus_worker, [(sets_dict, set_key) for set_key in sets_dict.keys()])
    return results

### 60 seconds!!! GPT insane; 5332 seconds on 40_40_40 dataset

example1 = "random_sets/10_10_10.tsv"
run_naive_sus(example1)
example2 = "random_sets/20_20_20.tsv"
run_naive_sus(example2)
example3 = "random_sets/30_30_30.tsv"
run_naive_sus(example3)
example4 = "random_sets/40_40_40.tsv"
run_naive_sus(example4)
# example5 = "random_sets/50_50_50.tsv"
# run_naive_sus(example5)
# example6 = "random_sets/60_60_60.tsv"
# run_naive_sus(example6)
# example7 = "random_sets/70_70_70.tsv"
# run_naive_sus(example7)

