from typing import Any, Iterator, Tuple
from lib.tools import *
import yaml
import os
from box import Box


"""
USAGE:
config, credentials, environment = get_config_credentials_environment()
settings = Settings(config)
-or-
settings = Box(config)
"""


class Settings():
    def __init__(self, config: dict):
        self.config = config

    def to_dict(self) -> dict:
        return self.config

    def g(self, path: str, default=None) -> Any:
        return g(self.config, path, default=default, sep='.')

    def get(self, path: str, default=None) -> Any:
        return g(self.config, path, default=default)

    def __getitem__(self, path) -> Any:
        default = None
        if isinstance(path, tuple):
            path, default = path
            
        v = g(self.config, path, default=default, sep='.')
        if isinstance(v, dict):
            v = Settings(v)
        return v

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self.config)
    
    def items(self):
        """Return an iterator of (key, value) pairs from the config dictionary."""
        for k, v in self.config.items():
            if isinstance(v, dict):
                yield k, Settings(v)
            else:
                yield k, v





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

    if not a:
        return c, environment    # 

    if not c:
        return a, environment    # 

    # Deep merge instead of shallow merge
    config = deep_merge(a, c)

    return config, environment


def get_config_credentials_environment(config_path='config.yaml', credentials_path='credentials.yaml'):
    """Get config, credentials, and environment with proper deep merging."""
    config = getYaml(config_path)

    if credentials_path:
        credentials = getYaml(credentials_path)
    else:
        credentials = {}
    
    # Deep-merge the credentials into the config
    config = deep_merge(config, credentials)
    
    env, environment = get_environment(config)
    a = g(config, 'all', {}) or {}
    c = g(config, env, {}) or {}
    
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