#include "cppgrad.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>

std::map<size_t, std::function<void()>> build_graph(Tensor& root) {
    std::thread::id tid = std::this_thread::get_id();

    // Map from each node ID to its list of children
    std::unordered_map<size_t, std::vector<size_t>> children;
    std::set<size_t> visited;
    std::queue<Tensor*> to_visit;

    // Initialize the queue with the root node
    to_visit.push(&root);

    // Track all nodes discovered in the first pass
    std::vector<Tensor*> all_nodes;

    while (!to_visit.empty()) {
        Tensor* curr = to_visit.front();
        to_visit.pop();

        // Skip if already visited
        if (visited.count(curr->id)) {
            continue;
        }
        visited.insert(curr->id);
        all_nodes.push_back(curr); // Store all discovered nodes

        // Iterate over parents and add to children map
        if (curr->parents.count(tid)) {
            for (auto& p : curr->parents[tid]) {
                children[p->id].push_back(curr->id);
                to_visit.push(p.get());  // Convert shared_ptr to raw pointer
            }
        }
    }

    /* Debug: Print children */
    for (auto& [id, child_ids] : children) {
        std::cout << "Node ID: " << id << " has children: ";
        for (auto& child_id : child_ids) {
            std::cout << child_id << " ";
        }
        std::cout << std::endl;
    }

    // Second pass: topological order
    std::set<size_t> visited_backwards;
    std::map<size_t, std::function<void()>> order;
    std::queue<Tensor*> to_visit_final;

    // Add all discovered nodes (not just root)
    for (Tensor* node : all_nodes) {
        to_visit_final.push(node);
    }

    while (!to_visit_final.empty()) {
        Tensor* curr = to_visit_final.front();
        to_visit_final.pop();

        // Skip if already visited
        if (visited_backwards.count(curr->id)) {
            continue;
        }

        // Check if all children have been processed
        bool ready = true;
        if (children.count(curr->id)) {  // Only check if this node has children
            for (auto& child_id : children[curr->id]) {
                if (visited_backwards.count(child_id) == 0) {
                    ready = false;
                    break;
                }
            }
        }

        if (ready) {
            visited_backwards.insert(curr->id);
            order[curr->id] = curr->backward_fn; // Store function
        } else {
            to_visit_final.push(curr);  // Retry later
        }
    }

    /* Debug: Print order from front to back */
    std::cout << "Execution Order (Backprop):" << std::endl;
    for (auto& [id, fn] : order) {
        std::cout << "Node ID: " << id << " has backward function" << std::endl;
    }

    return order;
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
    if (this->numel(shape) != 1) {
        throw std::runtime_error("backward() only supported for scalar outputs");
    }
    if (!requires_grad) {
        throw std::runtime_error("Tensor does not require gradients");
    }

    std::thread::id tid = std::this_thread::get_id();
    
    // Initialize *this* gradient to 1.0 for the backward start
    {
        std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
        if (!this->thread_gradients[tid]) {
            this->thread_gradients[tid] = std::make_shared<Tensor>(
                std::vector<float>(this->data.size(), 1.0f),
                this->shape, /*requires_grad=*/false
            );
        } else {
            // If gradient memory is already allocated, fill with 1.0
            std::fill(this->thread_gradients[tid]->data.begin(),
                      this->thread_gradients[tid]->data.end(),
                      1.0f);
        }
    }

    std::map<size_t, std::function<void()>> order = build_graph(*this);

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (it->second)
            it->second(); // Call the stored backward function
    }
}