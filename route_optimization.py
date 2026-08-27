# Illustrative Vehicle Routing Problem using Google OR-Tools.
# Replace the example distance matrix and demands with real logistics data.

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

distance_matrix = [
    [0, 10, 15, 20],
    [10, 0, 12, 18],
    [15, 12, 0, 8],
    [20, 18, 8, 0],
]

demands = [0, 1, 1, 1]
vehicle_capacity = 3
num_vehicles = 1
depot = 0

manager = pywrapcp.RoutingIndexManager(
    len(distance_matrix),
    num_vehicles,
    depot
)

routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return distance_matrix[from_node][to_node]

transit_callback = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return demands[from_node]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,
    [vehicle_capacity],
    True,
    "Capacity"
)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

solution = routing.SolveWithParameters(search_parameters)

if solution:
    index = routing.Start(0)
    route = []

    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    route.append(manager.IndexToNode(index))
    print("Optimized route:", route)
else:
    print("No feasible route found.")
