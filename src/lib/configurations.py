from lib.tools import *
import yaml
import os


"""
USAGE:
config, credentials, environment = get_config_credentials_environment()
"""


def get_environment(config='config.yaml'):
    """
    Given a configuration file, determine the environment based on the first environment that matches the environment test.
    @config_name: The name of the configuration file. Defaults to 'config'.
    @return: The environment and the configuration.
    @example:
        env, environment = get_environment('config')
        print(env)
        print(environment)
    """

    if isinstance(config, str):
        path = findPath(config)
        config = getYaml(path)
    env = None

    # Determine the environment by looking at each environment and seeing if the environment test matches the environment variables.
    if not 'environments' in config:
        raise ValueError("Configuration file does not contain an 'environments' section.")

    for key, environment_ in config['environments'].items():
        # Does it contain a test?
        test = environment_.get('test', None)
        if test:
            # Yes. Is there an environment variable?
            envvar = test.get('environment_variable', None)
            if envvar:
                # Yes. Is it set to the correct value?
                envvar_name = envvar.split('=')[0]
                envvar_value = envvar.split('=')[1]
                print(f"envvar_name: {envvar_name}, envvar_value: {envvar_value}")
                print(f"os.environ[envvar_name]: {os.environ.get(envvar_name)}")
                if envvar_name in os.environ and os.environ[envvar_name] == envvar_value:
                    # Yes. Set the environment and configuration.
                    env = key
                    environment = environment_
                    alias = environment_.get('alias', None)
                    port = environment.get('port', 5000)
                    if alias:
                        env = alias
                        environment = config['environments'][env]
                    break

    return env, environment



def get_config(config_path='config.yaml'):
    # OBSOLETE - use get_config_environment instead
    config = getYaml(config_path)
    env, environment = get_environment(config)
    a = g(config, 'all', {}) or {}
    c = g(config, env, {}) or {}
    return {**a, **c}


def get_config_environment(config_path='config.yaml', credentials_path='credentials.yaml'):
    config = getYaml(config_path)
    credentials = getYaml(credentials_path)

    # Deep-merge the credentials into the config by creating or overwriting the config with the credentials.
    config = deep_merge(config, credentials)

    env, environment = get_environment(config)
    a = g(config, 'all', {})
    c = g(config, env, {})

    # Deep merge instead of shallow merge
    config = deep_merge(a, c)

    return config, environment


def get_config_credentials_environment(config_path='config.yaml', credentials_path='credentials.yaml'):
    """Get config, credentials, and environment with proper deep merging."""
    config = getYaml(config_path)
    credentials = getYaml(credentials_path)
    
    # Deep-merge the credentials into the config
    config = deep_merge(config, credentials)
    
    env, environment = get_environment(config)
    a = g(config, 'all', {})
    c = g(config, env, {})
    
    # Deep merge instead of shallow merge
    config = deep_merge(a, c)
    
    return config, credentials, environment



def test1():
    s = """
      environments:
        prod:
          test:
            environment_variable: OSTYPE=linux-gnu
            port: 80
        local:
          test:
            environment_variable: "HOMEDRIVE=C:"
            port: 5000
    """
    config = yaml.safe_load(s)
    env, config = get_environment(config)
    print(env)
    print(config)


def test2():
    a = {
        'a': 1,
        'b': 2,
        'c': {
            'd': 3,
            'f': 4
        }
    }

    b = {
        'a': 10,
        'b': 20,
        'c': {
            'd': 30,
            'e': 40
        }
    }
    c = deep_merge(a, b)
    print(c)
    pass
    

if __name__ == "__main__":
    test1()
    test2()