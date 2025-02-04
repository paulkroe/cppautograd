#include "cppgrad.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>

/**
 * @brief Constructs a topologically sorted computation graph for backpropagation.
 *
 * This function traverses the computation graph starting from the given root tensor
 * and constructs a directed acyclic graph representing dependencies between tensors.
 * It orders the nodes in a valid topological sequence to ensure that gradients are propagated
 * in the correct order during backpropagation.
 *
 * The function follows these steps:
 * 1. **Graph Traversal:** Gathers all nodes in the computation graph from `root` to all its parents.
 * 2. **In-degree Calculation:** Computes the number of children for each node.
 * 3. **Topological Sorting:** Uses a queue to process nodes with in-degree = 0 first.
 * 4. **Backward Function Collection:** Stores backward functions in topological order.
 *
 * @param root The root tensor (i.e., the output tensor from which gradients will be propagated).
 * @return std::map<size_t, std::function<void()>> A map of tensor IDs to their respective
 *         backward functions, sorted in topological order.
 *
 * @note This function is thread-safe and associates each thread with its own computation graph.
 */
std::map<size_t, std::function<void()>> build_graph(Tensor& root) {
    std::thread::id tid = std::this_thread::get_id();

    /* 1. Gather the graph by traversing from root up to all parents */
    std::unordered_map<size_t, std::vector<size_t>> children;  // parent -> list of child IDs
    std::set<size_t> visited;
    std::queue<std::shared_ptr<TensorData>> to_visit;

    to_visit.push(root.ptr);
    std::vector<std::shared_ptr<TensorData>> all_nodes;

    while (!to_visit.empty()) {
        auto curr = to_visit.front();
        to_visit.pop();

        if (visited.count(curr->id)) {
            continue;
        }
        visited.insert(curr->id);
        all_nodes.push_back(curr);

        if (curr->parents.count(tid)) {
            for (auto& p : curr->parents[tid]) {
                children[p->id].push_back(curr->id);
                to_visit.push(p);
            }
        }
    }

    /* 2. Compute in-degrees of each node (how many children does each node have?) */
    /*    The "children" map is:  parentID -> listOfChildIDs */
    std::unordered_map<size_t,int> in_degree;
    // Initialize in_degree of every discovered node to 0
    for (auto node : all_nodes) {
        in_degree[node->id] = 0;
    }
    // Count the number of times each node appears as a child
    for (auto& kv : children) {
        for (auto& child_id : kv.second) {
            in_degree[child_id]++;
        }
    }

    /* 3. Initialize a queue with all nodes that have in-degree = 0 */
    std::queue<std::shared_ptr<TensorData>> ready;
    /* For quick ID->pointer lookup: */
    std::unordered_map<size_t, std::shared_ptr<TensorData>> id_to_tensor;
    for (auto node : all_nodes) {
        id_to_tensor[node->id] = node;
        /* if in degree is zero, node is ready to be processed */
        if (in_degree[node->id] == 0) {
            ready.push(node);
        }
    }

    /* 4. Pop from 'ready' and decrease the in-degree of children */
    std::map<size_t, std::function<void()>> topo_order;
    while (!ready.empty()) {
        auto curr = ready.front();
        ready.pop();

        /* Record the backward function in the topological order */
        topo_order[curr->id] = curr->backward_fn;

        /* For each child of curr, decrement the child's in-degree */
        if (children.count(curr->id)) {
            for (auto& child_id : children[curr->id]) {
                if (--in_degree[child_id] == 0) {
                    ready.push(id_to_tensor[child_id]);
                }
            }
        }
    }

    return topo_order;
}

/**
 * @brief Computes gradients for scalar tensors using backpropagation.
 *
 * This function performs automatic differentiation by propagating gradients backward
 * through the computation graph. It follows these steps:
 *
 * 1. **Validation:** Ensures that the tensor is scalar (`numel() == 1`) and requires gradients.
 * 2. **Gradient Initialization:** Sets the gradient of the tensor to `1.0`, which serves
 *    as the starting point for backpropagation.
 * 3. **Graph Construction:** Calls `build_graph()` to obtain a topologically sorted
 *    execution order for backpropagation.
 * 4. **Gradient Propagation:** Iterates over the nodes in reverse topological order,
 *    calling each stored backward function.
 *
 * @throws std::runtime_error If the tensor is not scalar or does not require gradients.
 *
 * @note This function is thread-safe and ensures that per-thread gradient tracking
 *       is correctly initialized before execution.
 *
 * @example
 * @code
 * Tensor loss = compute_loss(predictions, targets);
 * loss.backward(); // Computes gradients for all tensors involved in 'loss'
 * @endcode
 */
void Tensor::backward() {
    if (numel(ptr->shape) != 1) {
        throw std::runtime_error("backward() only supported for scalar outputs");
    }
    if (!ptr->requires_grad) {
        throw std::runtime_error("Tensor does not require gradients");
    }

    std::thread::id tid = std::this_thread::get_id();
    
    // Initialize *this* gradient to 1.0 for the backward start
    {
        std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
        if (!ptr->thread_gradients[tid]) {
                ptr->thread_gradients[tid] = std::make_shared<TensorData>(
                std::vector<float>(ptr->data.size(), 1.0f),
                ptr->shape, /*requires_grad=*/false
            );
        } else {
            // If gradient memory is already allocated, fill with 1.0
            std::fill(ptr->thread_gradients[tid]->data.begin(),
                      ptr->thread_gradients[tid]->data.end(),
                      1.0f);
        }
    }

    std::map<size_t, std::function<void()>> order = build_graph(*this);
    
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (it->second)
            it->second(); // Call the stored backward function
    }
}