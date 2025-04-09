# Dynamic Memory Allocation Simulation with Reinforcement Learning

This Streamlit application simulates different dynamic memory allocation algorithms, including First Fit, Best Fit, Worst Fit, Second Fit, and a Reinforcement Learning (RL) based approach. It visualizes the memory state after each allocation using bar charts and displays the calculated external fragmentation.

https://rl-mem-alloc-am-sr.streamlit.app/

## Overview

The application allows users to:

* Set the size of the memory.
* Input a sequence of process sizes to be allocated.
* Observe the initial state of the memory (with some pre-allocated blocks).
* Compare how different allocation algorithms place these processes in memory.
* Visualize the final memory layout for each algorithm as a bar chart, where allocated blocks are colored and hatched, and free blocks are light gray.
* See the calculated external fragmentation for each algorithm.
* Utilize a pre-trained Reinforcement Learning agent to perform memory allocation and compare its performance.

## Libraries and Imports

The application utilizes the following Python libraries:

* `streamlit`: For creating the interactive web application.
* `numpy`: For numerical operations and memory representation.
* `random`: For generating random process sizes and deallocation choices (though not directly used in the main allocation loop in this version).
* `torch`: For the Reinforcement Learning agent's neural network.
* `torch.nn`: Neural network modules for PyTorch.
* `torch.optim`: Optimization algorithms for PyTorch.
* `torch.nn.functional`: Functional interface for neural network operations.
* `collections.deque`: For the RL agent's experience replay buffer.
* `matplotlib.pyplot`: For creating the memory visualization bar charts.
* `os`: For checking the existence of the pre-trained RL agent weights file.

## Class Creations

The application defines the following classes:

* `MemoryAllocatorEnvironment`: Simulates the memory environment, handling allocation, deallocation, state representation, and reward calculation for the RL agent.
* `QNetwork`: Defines the neural network architecture for the RL agent's Q-function.
* `DQNAgent`: Implements the Deep Q-Network (DQN) agent, including acting, learning, and memory management.

## Memory Allocation Algorithms

The following classical memory allocation algorithms are implemented:

* **First Fit:** Allocates the first available free block that is large enough.
* **Best Fit:** Allocates the smallest available free block that is large enough.
* **Worst Fit:** Allocates the largest available free block.
* **Second Fit:** Allocates the second available free block that is large enough.

A Reinforcement Learning-based allocation is also implemented using a pre-trained `DQNAgent`.

## External Fragmentation Calculation

The `calculate_external_fragmentation_accurate` function calculates the external fragmentation of the memory after allocation.

## Visualisation

The `visualise_memory` function uses `matplotlib` to create a horizontal bar chart representing the memory. Free blocks are shown in light gray, and allocated blocks are colored and hatched to distinguish them.

## Usage

1.  Save the code as a Python file (e.g., `app.py`).
2.  Ensure you have all the required libraries installed (`pip install -r requirements.txt`).
3.  Make sure you have a pre-trained Reinforcement Learning agent weights file named `memory_allocator_dqn.pth` in the same directory as your `app.py` (or update the `load_path` variable accordingly).
4.  Run the Streamlit application from your terminal using the command: `streamlit run app.py`
5.  Interact with the application in your web browser:
    * Adjust the memory size using the slider in the sidebar.
    * Enter the sizes of the processes you want to allocate in the text input field in the sidebar (comma-separated). A suggested sequence is provided.
    * Observe the initial memory state.
    * View the memory allocation and fragmentation results for each algorithm below the input.

## Observations

The application prints the suggested process allocation sequence to the terminal. The results for each allocation algorithm, including the final memory visualization and the calculated external fragmentation, are displayed in the Streamlit app. The Reinforcement Learning agent's allocation is also shown if the pre-trained weights file is found.
