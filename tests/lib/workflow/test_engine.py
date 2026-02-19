import time
from lib.tools import *
from lib.workflow.engine import Engine




def test_light_switch():
    config = getYaml('tests/data/workflows/light-switch.yaml')
    environment = {
        'switch': False
    }

    def callback(engine, transition=None):
        if transition:
            # The engine wants to make this transition. We are given an opportunity to veto it.
            return transition # Allow it
        
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



if __name__ == '__main__':
    test_light_switch()