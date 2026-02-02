"""
This module implements a workflow engine.
The engine is created with an environent (dict), and a configuration object model, defined by a YAML file.
The callback is a function that is called when a transition is triggered. This gives you an optionaly hook into the engine to execute the transition.
If can be trivial in that it always returns True, or it can be more complex in that it executes the transition and returns True or False.

The environment can change overtime to represent the state of the world.
When tick() is called, the engine uses the environment to check if any transitions are ready to be triggered.
If so, then it makes the transition and sets the current state to the state of the transition.
"""




import time
from lib.tools import *


def sample_callback(engine, transition):
    if transition:
        # The engine wants to make this transition. We are given an opportunity to veto it.
        return transition # Allow it
        return None # Veto it. Leave the current state as is.
    
    # The engine is giving us a chance to select the next state.
    return None # Let the engine decide.




class Engine(object):
    def LoadEngine(filepath, callback=sample_callback, trace=None):
        engine_state = getYaml(filepath)
        engine = Engine(engine_state['config'], engine_state['environment'], callback, trace)
        engine.set_state(engine_state['last_state'])
        return engine
    
    def __init__(self, config, environment={}, callback=sample_callback, trace=None):
        """
        @config: A dictionary containing the workflow configuration.
        @environment: A dictionary containing the environment variables.
        @callback: A function that is called when a transition is triggered.
        @trace: A function(string) that is called to trace the engine.
        """
        self.callback = callback
        self.environment = environment
        self.config = config
        self.states = config['states']
        self.set_state('start')
        self.trace = trace

    def get_current_state(self):
        return self.states[self.current_state]

    def set_state(self, state_id):
        self.current_state = state_id
        state = self.states[state_id]
        state['id'] = state_id
        state['start_time'] = time.time()
        self.process_state(state)

    def load_states(self, filepath):
        engine_state = getYaml(filepath)
        last_state = engine_state['last_state']
        self.set_state(last_state)

    def save_states(self, filepath):
        # Helper function to recursively clean custom attributes from nested dicts
        def clean_dict(obj):
            """Remove custom attributes (id, start_time, end_time, duration) from dict and nested dicts."""
            if isinstance(obj, dict):
                cleaned = {}
                for key, value in obj.items():
                    # Skip custom runtime attributes
                    if key in ['id', 'start_time', 'end_time', 'duration']:
                        continue
                    # Recursively clean nested dicts and lists
                    cleaned[key] = clean_dict(value)
                return cleaned
            elif isinstance(obj, list):
                return [clean_dict(item) for item in obj]
            else:
                return obj
        
        # Create a clean copy of config
        clean_config = clean_dict(self.config)
        
        engine_state = {
            'config': clean_config,
            'environment': self.environment,
            'last_state': self.get_current_state()['id'],
        }
        writeYaml(filepath, engine_state)


    def process_state(self, state):
        if self.callback:
            newState = self.callback(self, state)
            if newState:
                state = newState
        state['end_time'] = time.time()
        state['duration'] = state['end_time'] - state['start_time']
        return state


    def tick(self):
        # Check if any transitions are ready to be triggered.
        # If so, trigger the transition and set the current state to the state of the transition.
        # Return False if no transition was triggered. Else return True.
        state = self.get_current_state()

        if not 'transitions' in state:
            # This must be a terminal state, so stop here.
            return False

        default_transition = None
        for transition_key, transition in state['transitions'].items():
            if 'when' in transition:
                all = True
                for wk, wv in transition['when'].items():
                    if wk == 'elapsed':
                        if time.time() - state['start_time'] >= to_seconds(wv):
                            continue
                        else:
                            all = False
                            break
                    if wk not in self.environment:
                        continue
                    if self.environment[wk] != wv:
                        all = False
                        break
                if all:
                    # Is there a hook?
                    if self.callback:
                        # Yes. Call it.
                        transition = self.callback(self, transition)
                    self.set_state(transition['state'])
                    return True
            else:
                if not default_transition:
                    default_transition = transition
        if default_transition:
            if self.callback:
                # Yes. Call it.
                transition = self.callback(self, default_transition)
                if transition:
                    self.set_state(transition['state'])
                    return True
            return True
        return False


    def run(self, delay_seconds=0.1):
        while True:
            if not self.tick():
                break
            time.sleep(delay_seconds)




# ================================================================================
# TESTS

def test_light_switch():
    config = getYaml('tests/data/workflows/light-switch.yaml')
    environment = {
        'switch': False
    }

    def callback(engine, transition=None):
        if transition:
            # The engine wants to make this transition. We are given an opportunity to veto it.
            return transition # Allow it
            return None # Veto it. Leave the current state as is.
        
        # The engine is giving us a chance to select the next state.
        return None # Let the engine decide.


    engine = Engine(config['light-switch'], environment, callback)
    assert engine.get_current_state()['id'] == 'start'
    assert engine.environment['switch'] == False

    # Someone flips the switch to on.
    environment['switch'] = True
    engine.tick()
    assert engine.get_current_state()['id'] == 'on'

    # Someone flips the switch to on again, even though it's already on.
    environment['switch'] = True
    engine.tick()
    assert engine.get_current_state()['id'] == 'on'
    assert engine.environment['switch'] == True

    # Nothing changed.
    engine.tick()
    assert engine.get_current_state()['id'] == 'on'
    assert engine.environment['switch'] == True

    # Someone flips the switch to off.
    environment['switch'] = False
    engine.tick()
    assert engine.get_current_state()['id'] == 'off'
    assert engine.environment['switch'] == False

    engine.save_states('tests/data/workflows/light-switch-engine-state.yaml')



def test_restore_engine_state():
    engine = Engine.LoadEngine('tests/data/workflows/light-switch-engine-state.yaml')
    assert engine.get_current_state()['id'] == 'off'
    assert engine.environment['switch'] == False



def test_engine_delay():
    states = """
    states:
      start:
        transitions:
          next:
            when:
              elapsed: 2s
            state: start
    """
    config = parseYaml(states)
    engine = Engine(config, environment={}, callback=None)
    engine.tick()
    time.sleep(2)
    engine.tick()


if __name__ == "__main__":
    test_engine_delay()
    test_light_switch()
    test_restore_engine_state()