#include "cppgrad.h"
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
void Tensor::backward(const size_t num_threads) {

    /* check if target is a scalar */
    if (this->numel(shape) != 1) {
        throw std::runtime_error("Backward only supported for scalar outputs");
    }

    /* check if target requires its gradient */
    if (!requires_grad) {
        throw std::runtime_error("This tensor does not require gradient");
    }

    /* initialize gradient as 1.0 */
    if (!grad) {
        grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 1.0f), shape, false);
    } else {
        std::fill(grad->data.begin(), grad->data.end(), 1.0f);
    }

    /* set of pointers to visited tensors */
    std::unordered_set<Tensor*> visited;
    /* set of tensor ids that have been visited */
    std::unordered_set<int> visited_id;
    /* stack to keep track of unprocessed tensors */
    std::stack<Tensor*> stack;
    /* initially stack only contains target */
    stack.push(this);

    while (!stack.empty()) {
        Tensor* current = stack.top();
        stack.pop();

        /* skip processing if already visited */
        if (visited_id.count(current->id)) continue;
        /* keep track of visited pointers */
        visited.insert(current);
        visited_id.insert(current->id);

        /* ensure current has gradient storage */
        if (!current->grad) {
            current->grad = std::make_shared<Tensor>(
                std::vector<float>(current->data.size(), 0.0f), current->shape, false
            );
        }

        // Execute the backward function if it exists
        if (current->backward_fn) {
            current->backward_fn(num_threads);
        }

        /* push parent nodes onto the stack, using raw pointers */
        for (const auto& parent : current->parents) {
            stack.push(parent.get()); /* push raw pointer on stack instead of shared_ptr */
        }
    }
}
