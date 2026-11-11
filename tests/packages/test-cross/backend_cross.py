import json
import os
import zipfile


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_host_requires_for_cross_build_wheel(config_settings=None, build_details=None):
    return ['host-extra == 1.0']


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = 'test_cross-1.0.0-py3-none-any.whl'
    with zipfile.ZipFile(os.path.join(wheel_directory, name), 'w') as whl:
        whl.writestr('test_cross-1.0.0.dist-info/METADATA', 'Metadata-Version: 2.1\nName: test-cross\nVersion: 1.0.0\n')
    return name


def cross_build_wheel(
    wheel_directory,
    config_settings=None,
    metadata_directory=None,
    build_details=None,
    host_requires_directory=None,
    toolchain=None,
):
    platform = build_details['platform'] or build_details['implementation']['_multiarch']
    name = f'test_cross-1.0.0-cp{build_details["language"]["version"].replace(".", "")}-none-{platform.replace("-", "_")}.whl'
    with zipfile.ZipFile(os.path.join(wheel_directory, name), 'w') as whl:
        whl.writestr('test_cross-1.0.0.dist-info/METADATA', 'Metadata-Version: 2.1\nName: test-cross\nVersion: 1.0.0\n')
        whl.writestr(
            'cross.json',
            json.dumps(
                {
                    'build_details': build_details,
                    'host_requires_directory': host_requires_directory,
                    'host_requires_contents': sorted(os.listdir(host_requires_directory)) if host_requires_directory else None,
                    'toolchain': toolchain,
                    'config_settings': config_settings,
                }
            ),
        )
    return name
