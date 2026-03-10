# Workflow Engine


The workflow engine is a simple state machine that can 


```python
from lib.workflow.engine import Engine


# Read in some workflow definition
config = getYaml('tests/data/workflows/light-switch.yaml')
config = config['light-switch']


# Define an environment, which can be built from a database or some external watching system.
environment = {
    'switch': False
}


# Define an optional callback function for hooking into the transition events.
def callback(engine, transition=None):
    if transition:
        # The engine wants to make this transition. We are given an opportunity to veto it.
        return transition # Allow it
    
    # The engine is giving us a chance to select the next state.
    return None # Let the engine decide.


# Create an engine object from the config definition, the environment, and the optional callback.
engine = Engine(config, environment, callback)


# Call tick() when the environment changes, or each time you want the workflow to react to the environment based on its current state.
engine.tick()


# Print the current state
print(engine.get_current_state()['id'])


# Save the state of the workflow system, which includes the workflow, the entire environemnt state, and the current workflow state.
engine.save_states('tests/data/workflows/light-switch-engine-state.yaml')


# Restore the saved state.
engine = Engine.LoadEngine('tests/data/workflows/light-switch-engine-state.yaml')

```



