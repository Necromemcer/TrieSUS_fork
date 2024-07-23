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

    # Create the solution collector
    solution_collector = SolutionCollector(set_vars)

    # Solve the model with the solution callback
    status = solver.SolveWithSolutionCallback(model, solution_collector)

    # Return all collected solutions and the status
    return solution_collector.solutions, status