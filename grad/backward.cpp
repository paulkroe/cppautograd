#include "cppgrad.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>

std::map<size_t, std::function<void()>> build_graph(Tensor& root) {
    std::thread::id tid = std::this_thread::get_id();

    // 1. Gather the graph by traversing from root up to all parents
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
                // p is a shared_ptr<Tensor>
                children[p->id].push_back(curr->id);
                to_visit.push(p);
            }
        }
    }

    // 2. Compute in-degrees of each node (how many children does each node have?)
    //    The "children" map is:  parentID -> listOfChildIDs
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

    // 3. Initialize a queue with all nodes that have in-degree = 0
    std::queue<std::shared_ptr<TensorData>> ready;
    // For quick ID->pointer lookup:
    std::unordered_map<size_t, std::shared_ptr<TensorData>> id_to_tensor;
    for (auto node : all_nodes) {
        id_to_tensor[node->id] = node;
        if (in_degree[node->id] == 0) {
            ready.push(node);
        }
    }

    // 4. Pop from 'ready' and decrease the in-degree of children
    std::map<size_t, std::function<void()>> topo_order;
    while (!ready.empty()) {
        auto curr = ready.front();
        ready.pop();

        // Record the backward function in the topological order
        topo_order[curr->id] = curr->backward_fn;

        // For each child of curr, decrement the child's in-degree
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

/* 
 * backward function for scalar tensors:
 * 1. assert target tensor is scalar and requires its gradient
 * 3. Initialize/Set the gradient to 1.0
 * 4. initialize a stack of tensors that need to be processed
 * 5. while stack is not empty:
 *     a. pop the top tensor from the stack
 *     b. if tensor has been visited, skip processing
 *     c. if tensor has a backward function, execute it
 *     d. push parent nodes onto the stack
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