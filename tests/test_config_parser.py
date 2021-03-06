import os

from numpy.testing import assert_raises

from blocks.config_parser import Configuration, ConfigurationError


def test_config_parser():
    _environ = dict(os.environ)
    os.environ['BLOCKS_CONFIG'] = os.path.join(os.getcwd(),
                                               '.test_blocksrc')
    with open(os.environ['BLOCKS_CONFIG'], 'w') as f:
        f.write('data_path: yaml_path')
    if 'BLOCKS_DATA_PATH' in os.environ:
        del os.environ['BLOCKS_DATA_PATH']
    try:
        config = Configuration()
        config.add_config('data_path', str, env_var='BLOCKS_DATA_PATH')
        config.add_config('config_with_default', int, default='1',
                          env_var='BLOCKS_CONFIG_TEST')
        config.add_config('config_without_default', str)
        assert config.data_path == 'yaml_path'
        os.environ['BLOCKS_DATA_PATH'] = 'env_path'
        assert config.data_path == 'env_path'
        assert config.config_with_default == 1
        os.environ['BLOCKS_CONFIG_TEST'] = '2'
        assert config.config_with_default == 2
        assert_raises(ConfigurationError, getattr, config,
                      'non_existing_config')
        assert_raises(ConfigurationError, getattr, config,
                      'config_without_default')
    finally:
        os.remove(os.environ['BLOCKS_CONFIG'])
        os.environ.clear()
        os.environ.update(_environ)
