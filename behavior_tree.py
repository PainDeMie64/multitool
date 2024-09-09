"""
=============
Behavior Tree
=============

Implementing a simple behavior tree with a sequence of actions 
for a game bot.
"""

"""
------------------
Core functionality
------------------
"""

class BehaviorTree:
    def __init__(self):
        self.root = None

    def set_root(self, node):
        self.root = node

    def run(self):
        if self.root:
            return self.root.run()

class Node:
    def run(self):
        pass

class Sequence(Node):
    def __init__(self, children):
        self.children = children

    def run(self):
        return all(child.run() for child in self.children)

class Selector(Node):
    def __init__(self, children):
        self.children = children

    def run(self):
        return any(child.run() for child in self.children)

class Task(Node):
    def __init__(self, task_func):
        self.task_func = task_func

    def run(self):
        return self.task_func()

"""Example Tasks"""
def move_to_target():
    print("Moving to target...")
    return True

def attack_target():
    print("Attacking target...")
    return True

def target_in_range():
    print("Checking if target is in range...")
    return True

"""Build and run behavior tree"""
if __name__ == "__main__":
    # Create the behavior tree
    tree = BehaviorTree()

    # Create tasks
    move_task = Task(move_to_target)
    attack_task = Task(attack_target)
    check_range_task = Task(target_in_range)

    # Create sequences and selectors
    attack_sequence = Sequence([check_range_task, attack_task])
    move_and_attack_sequence = Sequence([move_task, attack_sequence])

    # Set the root of the tree
    tree.set_root(move_and_attack_sequence)

    # Run the tree
    tree.run()