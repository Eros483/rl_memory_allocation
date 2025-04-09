#all libraries and imports
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import streamlit as st

st.title("Memory Allocation Simulation")

#all class creations
class MemoryAllocatorEnvironment:
    def __init__(self, memory_size):
        self.memory_size=memory_size

        #creating the memory representation
        self.memory=np.zeros(memory_size, dtype=int)
        self.allocated_blocks={}
        self.next_process_id=1
        self.fragmentation_threshold=0.8

    def reset(self, initial_memory=None, isBlank=False):
        # Hard-coded comprehensive initial memory
        self.memory = np.zeros(self.memory_size, dtype=int)
        self.memory[5:15] = 1
        self.memory[25:30] = 2
        self.memory[40:55] = 3
        self.memory[65:70] = 4
        self.memory[80:95] = 5

        if isBlank:
            self.memory = np.zeros(self.memory_size, dtype=int)

        self.allocated_blocks = {}
        # Initialize allocated blocks based on the hard-coded memory
        for i in np.unique(self.memory[np.where(self.memory > 0)]):
            indices = np.where(self.memory == i)[0]
            if indices.size > 0:
                start = indices[0]
                size = indices[-1] - indices[0] + 1
                self.allocated_blocks[i] = (start, size)
                self.next_process_id = max(self.next_process_id, i + 1)

        initial_request_size = self._generate_allocation_request()
        return self._get_state(initial_request_size)
    
    def _get_state(self, current_request_size):
        #a simple low cost state representation of memory
        memory_status=np.array(self.memory>0, dtype=int)
        return np.concatenate([memory_status, [current_request_size/self.memory_size]])
    
    def step(self, action, request_size=None):
        #allocates memory based on action
        if request_size is None:    
            request_size=self._generate_allocation_request()
        reward=0
        done=False
        info={}

        if 0<=action<=self.memory_size - request_size:
            if np.all(self.memory[action:action+request_size]==0):
                #allocating memory
                process_id = self.next_process_id
                self.memory[action : action + request_size] = process_id
                self.allocated_blocks[process_id]=(action, request_size)
                self.next_process_id+=1
                reward= 0.1  # Small positive reward for successful allocation
            else:
                reward= -1.0  # Negative reward for attempting to allocate in occupied space
        else:
            reward= -1.0

        free_blocks=np.where(self.memory==0)[0] #checking for fragmentation

        if len(free_blocks>0):
            max_free_block_size=0
            current_free_block_size=0
            for i in range(len(free_blocks)):
                if i==len(free_blocks)-1 or free_blocks[i+1]!=free_blocks[i]+1:
                    max_free_block_size=max(max_free_block_size, current_free_block_size)
                    current_free_block_size=0

            if max_free_block_size<5 and np.sum(self.memory==0)/self.memory_size<(1-self.fragmentation_threshold):
                #high fragmentation
                done=True
                reward-=0.5
                info['reason']='high_Fragmentation'

        elif np.all(self.memory>0):
            done=True
            info['reason']='memory_full'

        next_state=self._get_state(request_size)
        return next_state, reward, done, info
    
    def _generate_allocation_request(self):
        #generates memory allocation request
        return random.randint(1,10)
    
    def deallocate(self, process_id=None):
        #deallocates specific process
        if self.allocated_blocks:
            if process_id is None:
                process_id = random.choice(list(self.allocated_blocks.keys()))

        if process_id in self.allocated_blocks:
            start, size=self.allocated_blocks[process_id]
            self.memory[start:start+size]=0
            del self.allocated_blocks[process_id]

    def get_available_actions(self, request_size=None):
        #returns a list  of possible starting adresses for allocations
        if request_size is None:
            request_size=self._generate_allocation_request
        
        available_actions=[]
        for i in range(self.memory_size-request_size+1):
            if np.all(self.memory[i:i+request_size]==0):
                available_actions.append(i)

        return available_actions
        
    def render(self):
        #prints the current state of the memory
        memory_representation=['_' if cell==0 else str(cell) for cell in self.memory]
        print("Memory: ", "".join(memory_representation))
        print("\n")
        print("Allocated: ", self.allocated_blocks)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1=nn.Linear(state_size, hidden_size)
        self.fc2=nn.Linear(hidden_size, hidden_size)
        self.fc3=nn.Linear(hidden_size, action_size)

        self.reward_history=[]

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, buffer_size=10000, batch_size=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, device="cpu"):
        self.state_size=state_size
        self.action_size=action_size
        self.learning_rate=learning_rate
        self.gamma=gamma
        self.memory=deque(maxlen=buffer_size)
        self.batch_size=batch_size
        self.epsilon_start=epsilon_start
        self.epsilon=epsilon_start
        self.epsilon_end=epsilon_end
        self.epsilon_decay=epsilon_decay
        self.device=device

        self.q_network=QNetwork(state_size, action_size).to(self.device)
        self.target_network=QNetwork(state_size, action_size).to(self.device)
        self.optimizer=optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.update_target_network()

        self.reward_history=[]

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_actions, force_exploit=False):
        if force_exploit or random.random() >= self.epsilon:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            self.q_network.train()
            valid_q_values = q_values[0][available_actions]
            best_action_index = torch.argmax(valid_q_values).item()
            return available_actions[best_action_index]
        else:
            return random.choice(available_actions)
        
    def learn(self):
        if len(self.memory)<self.batch_size:
            return
        
        experiences=random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, done=zip(*experiences)

        state_tensor=torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_tensor=torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_tensor=torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        next_states_tensor=torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_tensor=torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values=self.q_network(state_tensor).gather(1, actions_tensor)

        next_q_values=self.target_network(next_states_tensor).max(1)[0].unsqueeze(1)
        targets=reward_tensor+self.gamma*(1-dones_tensor)*next_q_values

        loss=F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon=max(self.epsilon_end, self.epsilon*self.epsilon_decay)


def allocate_processes_with_agent(processes, memory_size, agent, use_blank_memory=False, render=True):
    env=MemoryAllocatorEnvironment(memory_size)
    state=env.reset(isBlank=use_blank_memory)
    allocation_map={}
    all_allocated=True
    final_memory_rr=env.memory
    agent.epsilon=0.0
    agent.q_network.eval()

    for i, process_size in enumerate(processes):
        available_actions=env.get_available_actions(request_size=process_size)

        if available_actions:
            action=agent.act(state, available_actions, force_exploit=True)
            next_state, reward, done, info = env.step(action, request_size=process_size)
            allocation_map[i]=action
            state=next_state
            if render:
                print(f"Allocated process {i+1} of size: {process_size} at adress {action} [info: {info}]")
                env.render()
        else:
            print(f"Couldn't allocate process {i+1} of size: {process_size}. No available space.")
            all_allocated=False

    agent.q_network.train()
    
    final_memory_rr=env.memory

    return allocation_map, all_allocated, final_memory_rr

def calculate_external_fragmentation_accurate(memory):
    total_memory = len(memory)
    free_blocks = np.where(memory == 0)[0]
    total_free_space = len(free_blocks)

    if total_memory == 0:
        return 0.0

    if total_free_space == 0:
        return 0.0

    if total_free_space == total_memory:
        return 0.0  # No fragmentation if all memory is free

    contiguous_free_block_lengths = []
    if total_free_space > 0:
        current_length = 0
        for i in range(total_memory):
            if memory[i] == 0:
                current_length += 1
            else:
                if current_length > 0:
                    contiguous_free_block_lengths.append(current_length)
                    current_length = 0
        if current_length > 0:
            contiguous_free_block_lengths.append(current_length)

    num_contiguous_free_blocks = len(contiguous_free_block_lengths)

    if total_free_space > 0:

        fragmentation_ratio = num_contiguous_free_blocks / total_free_space if total_free_space > 0 else 0.0

        average_free_block_size = total_free_space / num_contiguous_free_blocks if num_contiguous_free_blocks > 0 else 0

        if average_free_block_size > 0:
            fragmentation_score = 1.0 - (average_free_block_size / total_free_space)
        else:
            fragmentation_score = 1.0 
        num_allocated_blocks = total_memory - total_free_space
        if num_allocated_blocks > 0:
            fragmentation_score_alt = num_contiguous_free_blocks / num_allocated_blocks
            # Normalize this to be between 0 and 1 (heuristically)
            fragmentation_score = fragmentation_score_alt / (fragmentation_score_alt + 1)
        else:
            fragmentation_score = 0.0 # No fragmentation if nothing is allocated

        if contiguous_free_block_lengths:
            variance = np.var(contiguous_free_block_lengths)
            # Normalize variance by total memory size (heuristic)
            fragmentation_score_variance = variance / total_memory
            # Combine with the ratio of free blocks
            fragmentation_score = (fragmentation_ratio + fragmentation_score_variance) / 2.0

        total_blocks = 1 if total_memory > 0 else 0 # Consider memory as one whole block if empty
        if total_blocks > 0:
            fragmentation_score_simple = num_contiguous_free_blocks / total_blocks
        else:
            fragmentation_score_simple = 0.0

        fragmentation_score = num_contiguous_free_blocks / total_memory if total_memory > 0 else 0.0


        return max(0.0, min(1.0, fragmentation_score))
    else:
        return 0.0

def first_fit(memory, process_size, process_id):
    """Allocates memory using the First Fit algorithm.

    Returns the modified memory array.
    """
    allocated_memory = np.copy(memory)
    for i in range(len(allocated_memory) - process_size + 1):
        if np.all(allocated_memory[i : i + process_size] == 0):
            allocated_memory[i : i + process_size] = process_id
            return allocated_memory
    return allocated_memory

def best_fit(memory, process_size, process_id):
    """Allocates memory using the Best Fit algorithm.

    Returns the modified memory array.
    """
    allocated_memory = np.copy(memory)
    best_fit_start = -1
    min_remaining_size = float('inf')

    for i in range(len(allocated_memory) - process_size + 1):
        if np.all(allocated_memory[i : i + process_size] == 0):
            # Calculate contiguous free block size
            current_free_block_size = 0
            j = i
            while j < len(allocated_memory) and allocated_memory[j] == 0:
                current_free_block_size += 1
                j += 1

            if current_free_block_size >= process_size:
                remaining_size = current_free_block_size - process_size
                if remaining_size < min_remaining_size:
                    min_remaining_size = remaining_size
                    best_fit_start = i

    if best_fit_start != -1:
        allocated_memory[best_fit_start : best_fit_start + process_size] = process_id
    return allocated_memory

def worst_fit(memory, process_size, process_id):
    """Allocates memory using the Worst Fit algorithm.

    Returns the modified memory array.
    """
    allocated_memory = np.copy(memory)
    worst_fit_start = -1
    max_remaining_size = -1

    for i in range(len(allocated_memory) - process_size + 1):
        if np.all(allocated_memory[i : i + process_size] == 0):
            # Calculate contiguous free block size
            current_free_block_size = 0
            j = i
            while j < len(allocated_memory) and allocated_memory[j] == 0:
                current_free_block_size += 1
                j += 1

            if current_free_block_size >= process_size:
                remaining_size = current_free_block_size - process_size
                if remaining_size > max_remaining_size:
                    max_remaining_size = remaining_size
                    worst_fit_start = i

    if worst_fit_start != -1:
        allocated_memory[worst_fit_start : worst_fit_start + process_size] = process_id
    return allocated_memory

def second_fit(memory, process_size, process_id):
    """Allocates memory using the Second Fit algorithm (finds the second suitable block).

    Returns the modified memory array.
    """
    allocated_memory = np.copy(memory)
    suitable_starts = []
    for i in range(len(allocated_memory) - process_size + 1):
        if np.all(allocated_memory[i : i + process_size] == 0):
            suitable_starts.append(i)

    if len(suitable_starts) >= 2:
        second_fit_start = suitable_starts[1]
        allocated_memory[second_fit_start : second_fit_start + process_size] = process_id
    return allocated_memory

memory_size = 100

def visualise_memory(memory, title):
    figure, ax=plt.subplots(figsize=(10, 1))
    ax.set_title(title)
    ax.set_xlim(0, len(memory))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Memory Address")

    start=0
    while start<len(memory):
        if memory[start]==0:
            end=start
            while end<len(memory) and memory[end]==0:
                end+=1
            ax.broken_barh([(start, end-start)], (0.4, 0.2), facecolors='lightgray', edgecolor='black')
            start=end
        
        else:
            process_id=memory[start]
            end=start
            while end<len(memory) and memory[end]==process_id:
                end+=1
            ax.broken_barh([(start, end-start)], (0.4, 0.2), facecolors=f'C{process_id%10}', edgecolor='black', hatch='///')
            start=end
    st.pyplot(figure)

print("Suggested process allocation for observations:  8, 12, 5, 9, 11, 4, 6")

initial_memory_comprehensive  = np.zeros(memory_size, dtype=int)
initial_memory_comprehensive[5:15] = 1  # Size 10
initial_memory_comprehensive[25:30] = 2  # Size 5
initial_memory_comprehensive[40:55] = 3  # Size 15
initial_memory_comprehensive[65:70] = 4  # Size 5
initial_memory_comprehensive[80:95] = 5  # Size 15

initial_memory=initial_memory_comprehensive 

st.subheader("Initial Memory with free blocks: ")
visualise_memory(initial_memory, "Initial Memory State")

process_sizes_str=st.sidebar.text_input("Enter the sizes of the processes to allocate (comma-seperated, suggested: 8, 12, 5, 9, 11, 4, 6): ")
if process_sizes_str:
    processes=[int(size.strip()) for size in process_sizes_str.split(",")]

    memory_ff=np.copy(initial_memory)
    memory_bf=np.copy(initial_memory)
    memory_wf=np.copy(initial_memory)
    memory_sf=np.copy(initial_memory)

    print("\nFirst Fit:")
    i=0
    for size in processes:
        i+=1

        memory_before_ff=np.copy(memory_ff)

        memory_ff=first_fit(memory_ff, size, i)

        allocated_indices=np.where((memory_ff!=memory_before_ff)&(memory_ff==i))[0]
        if allocated_indices.size>0:
            print(f"Process {i} allocated at address {allocated_indices[0]} with size {size}")
        else:
            print(f"Process {i} cannot be allocated by ff")

    print("\nBest Fit:")
    i=0
    for size in processes:
        i+=1

        memory_before_bf=np.copy(memory_bf)

        memory_bf=best_fit(memory_bf, size, i)

        allocated_indices=np.where((memory_bf!=memory_before_bf)&(memory_bf==i))[0]
        if allocated_indices.size>0:
            print(f"Process {i} allocated at address {allocated_indices[0]} with size {size}")
        else:
            print(f"Process {i} cannot be allocated by bf")

    print("\nWorst Fit:")
    i=0
    for size in processes:
        i+=1

        memory_before_wf=np.copy(memory_wf)

        memory_wf=worst_fit(memory_wf, size, i)

        allocated_indices=np.where((memory_wf!=memory_before_wf)&(memory_wf==i))[0]
        if allocated_indices.size>0:
            print(f"Process {i} allocated at address {allocated_indices[0]} with size {size}")
        else:
            print(f"Process {i} cannot be allocated by wf")

    print("\nSecond Fit:")
    i=0
    for size in processes:
        i+=1

        memory_before_sf=np.copy(memory_sf)

        memory_sf=second_fit(memory_sf, size, i)

        allocated_indices=np.where((memory_sf!=memory_before_sf)&(memory_sf==i))[0]
        if allocated_indices.size>0:
            print(f"Process {i} allocated at address {allocated_indices[0]} with size {size}")
        else:
            print(f"Process {i} cannot be allocated by sf")

    state_size=memory_size+1
    action_size=memory_size
    agent=DQNAgent(state_size, action_size)

    load_path="memory_allocator_dqn.pth"
    agent.q_network.load_state_dict(torch.load(load_path))
    print(f"loaded trained model from path: {load_path}")

    allocation_result, _, mem_rl=allocate_processes_with_agent(processes, memory_size, agent)

    print("\nReinforcement Learning Algorithm:")
    print(mem_rl)

    def render_memory_array(memory):
        memory_representation = ['_' if cell == 0 else str(cell) for cell in memory]
        print("Memory:", "".join(memory_representation))

    print("\nBest Fit")
    render_memory_array(memory_bf)
    a=calculate_external_fragmentation_accurate(memory_bf)
    print("Fragmentation in best fit: ", a)

    st.subheader("Best Fit allocation: ")
    visualise_memory(memory_bf, f"Best fit (fragmentation: {a})")
    

    print("\nFirst Fit")
    render_memory_array(memory_ff)
    b=calculate_external_fragmentation_accurate(memory_ff)
    print("Fragmentation in first fit: ", b)

    st.subheader("First Fit allocation: ")
    visualise_memory(memory_ff, f"Best fit (fragmentation: {b})")

    print("\nSecond Fit")
    render_memory_array(memory_sf)
    c=calculate_external_fragmentation_accurate(memory_sf)
    print("Fragmentation in second fit: ", c)

    st.subheader("Second Fit allocation: ")
    visualise_memory(memory_sf, f"Best fit (fragmentation: {c})")

    print("\nWorst Fit")
    render_memory_array(memory_wf)
    d=calculate_external_fragmentation_accurate(memory_wf)
    print("Fragmentation in worst fit: ", d)

    st.subheader("Worst Fit allocation: ")
    visualise_memory(memory_wf, f"Best fit (fragmentation: {d})")

    print("\nReinforcement Learning Algorithm")
    render_memory_array(mem_rl)
    e=calculate_external_fragmentation_accurate(mem_rl)
    print("Fragmentation in rl algorithm: ", e)

    st.subheader("Reinforcement learning allocation: ")
    visualise_memory(mem_rl, f"Best fit (fragmentation: {e})")

    #visualisations
