#!/usr/bin/env python3


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
            ### Keep only the shortest solutions
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