# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import zipfile

from typing import TYPE_CHECKING

import pytest
import pytest_mock

import build
import build.__main__
import build._cross

from build._cross import CrossBuildConfig


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager

    from build._types import JSONValue, TOMLValue


def make_build_details(
    platform: str = 'linux-aarch64',
    *,
    multiarch: str | None = 'aarch64-linux-gnu',
    extension_suffix: str = '.cpython-314-aarch64-linux-gnu.so',
    flags: list[str] | None = None,
    base_prefix: str = '/opt/host-python',
) -> dict[str, JSONValue]:
    details: dict[str, JSONValue] = {
        'schema_version': '1.0',
        'base_prefix': base_prefix,
        'base_interpreter': './bin/python3.14',
        'platform': platform,
        'language': {
            'version': '3.14',
            'version_info': {'major': 3, 'minor': 14, 'micro': 8, 'releaselevel': 'final', 'serial': 0},
        },
        'implementation': {
            'name': 'cpython',
            'cache_tag': 'cpython-314',
            'version': {'major': 3, 'minor': 14, 'micro': 8, 'releaselevel': 'final', 'serial': 0},
        },
        'abi': {'flags': flags or [], 'extension_suffix': extension_suffix, 'stable_abi_suffix': '.abi3.so'},
    }
    if multiarch is not None:
        implementation = details['implementation']
        assert isinstance(implementation, dict)
        implementation['_multiarch'] = multiarch
    return details


@pytest.fixture
def build_details_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / 'lib' / 'python3.14' / 'build-details.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(make_build_details(base_prefix='../..')), encoding='utf-8')
    return path


@pytest.fixture
def cross_config() -> CrossBuildConfig:
    return CrossBuildConfig(build_details=make_build_details(), toolchain={'CC': 'aarch64-linux-gnu-gcc'})


# -- pyproject.toml parsing ---------------------------------------------------


def test_parse_host_requires() -> None:
    table = build._builder._parse_build_system_table(
        {'build-system': {'requires': ['foo'], 'host-requires': ['numpy', 'cffi; os_name == "posix"']}}
    )
    assert table.get('host-requires') == ['numpy', 'cffi; os_name == "posix"']


def test_parse_host_requires_absent() -> None:
    table = build._builder._parse_build_system_table({'build-system': {'requires': ['foo']}})
    assert 'host-requires' not in table


@pytest.mark.parametrize('value', ['numpy', [1], {'a': 'b'}])
def test_parse_host_requires_invalid(value: TOMLValue) -> None:
    with pytest.raises(build.BuildSystemTableValidationError, match='`host-requires` must be an array of strings'):
        build._builder._parse_build_system_table({'build-system': {'requires': ['foo'], 'host-requires': value}})


def test_parse_host_requires_typo() -> None:
    with (
        pytest.warns(build.TypoWarning, match="did you mean 'host-requires'"),
        pytest.raises(build.BuildSystemTableValidationError, match='Unknown properties: host-require'),
    ):
        build._builder._parse_build_system_table({'build-system': {'requires': ['foo'], 'host-require': ['numpy']}})


def test_project_builder_host_requires(package_test_cross: str) -> None:
    builder = build.ProjectBuilder(package_test_cross)
    assert builder.host_requires == {'numpy', "pywin32; sys_platform == 'win32'", "cffi; platform_machine == 'aarch64'"}


def test_project_builder_host_requires_default(package_test_no_prepare: str) -> None:
    assert build.ProjectBuilder(package_test_no_prepare).host_requires == set()


# -- build details ------------------------------------------------------------


def test_load_build_details(build_details_file: pathlib.Path, tmp_path: pathlib.Path) -> None:
    details = build._cross.load_build_details(build_details_file)
    assert details['platform'] == 'linux-aarch64'
    # a relative base_prefix is resolved against the file location
    assert details['base_prefix'] == os.fspath(tmp_path)


def test_load_build_details_absolute_prefix(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'build-details.json'
    path.write_text(json.dumps(make_build_details(base_prefix='/opt/python')), encoding='utf-8')
    assert build._cross.load_build_details(path)['base_prefix'] == '/opt/python'


def test_load_build_details_missing(tmp_path: pathlib.Path) -> None:
    with pytest.raises(build.BuildException, match=r'build-details\.json not found'):
        build._cross.load_build_details(tmp_path / 'missing.json')


def test_load_build_details_invalid_json(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'build-details.json'
    path.write_text('{', encoding='utf-8')
    with pytest.raises(build.BuildException, match='Failed to parse'):
        build._cross.load_build_details(path)


@pytest.mark.parametrize(
    ('content', 'message'),
    [
        ('[]', 'does not contain a JSON object'),
        (
            '{"schema_version": "2.0", "language": {}, "implementation": {}, "abi": {}}',
            'unsupported build-details.json schema_version',
        ),
        ('{"schema_version": "1.0", "language": {}, "implementation": {}}', "missing required key 'abi'"),
    ],
)
def test_load_build_details_invalid(tmp_path: pathlib.Path, content: str, message: str) -> None:
    path = tmp_path / 'build-details.json'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(build.BuildException, match=message):
        build._cross.load_build_details(path)


def test_cross_config_properties(cross_config: CrossBuildConfig) -> None:
    assert cross_config.python_version == '3.14'
    assert cross_config.implementation == 'cpython'
    assert cross_config.platform_tags == ['linux_aarch64', 'manylinux_2_17_aarch64', 'manylinux2014_aarch64']
    assert cross_config.abis[:3] == ['cp314', 'abi3', 'none']
    assert cross_config.marker_environment['platform_machine'] == 'aarch64'
    assert cross_config.marker_environment['sys_platform'] == 'linux'


def test_cross_config_glibc_version() -> None:
    cross = CrossBuildConfig(build_details=make_build_details(), toolchain={}, glibc_version=(2, 28))
    assert cross.platform_tags[:3] == ['linux_aarch64', 'manylinux_2_28_aarch64', 'manylinux_2_27_aarch64']
    assert 'manylinux2014_aarch64' in cross.platform_tags


def test_cross_config_empty_platform() -> None:
    # python-build-standalone ships an empty "platform" key
    cross = CrossBuildConfig(build_details=make_build_details(''), toolchain={})
    assert cross.platform_tags[0] == 'linux_aarch64'
    assert cross.marker_environment['platform_system'] == 'Linux'


@pytest.mark.parametrize(
    ('value', 'expected'),
    [('2.28', (2, 28)), ('2.17', (2, 17))],
)
def test_parse_glibc_version(value: str, expected: tuple[int, int]) -> None:
    assert build._cross.parse_glibc_version(value) == expected


@pytest.mark.parametrize('value', ['2', '2.28.1', 'two.x', ''])
def test_parse_glibc_version_invalid(value: str) -> None:
    with pytest.raises(build.BuildException, match='Invalid glibc version'):
        build._cross.parse_glibc_version(value)


# -- toolchain ----------------------------------------------------------------


def test_toolchain_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in build._cross.TOOLCHAIN_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CC', 'aarch64-linux-gnu-gcc')
    monkeypatch.setenv('CFLAGS', '-O2')
    monkeypatch.setenv('PKG_CONFIG_LIBDIR', '/sysroot/usr/lib/pkgconfig')
    monkeypatch.setenv('UNRELATED', 'x')
    assert build._cross.toolchain_from_environment() == {
        'CC': 'aarch64-linux-gnu-gcc',
        'CFLAGS': '-O2',
        'PKG_CONFIG_LIBDIR': '/sysroot/usr/lib/pkgconfig',
    }
    assert build._cross.toolchain_from_environment({'CC': 'clang', 'AR': 'llvm-ar'}) == {
        'CC': 'clang',
        'CFLAGS': '-O2',
        'PKG_CONFIG_LIBDIR': '/sysroot/usr/lib/pkgconfig',
        'AR': 'llvm-ar',
    }


def test_parse_toolchain_args() -> None:
    assert build._cross.parse_toolchain_args(None) == {}
    assert build._cross.parse_toolchain_args(['CC=gcc', 'CFLAGS=-O2 -DX=1']) == {'CC': 'gcc', 'CFLAGS': '-O2 -DX=1'}


@pytest.mark.parametrize('arg', ['CC', '=gcc', ''])
def test_parse_toolchain_args_invalid(arg: str) -> None:
    with pytest.raises(build.BuildException, match='Invalid toolchain setting'):
        build._cross.parse_toolchain_args([arg])


# -- host requirements --------------------------------------------------------


def test_filter_host_requirements(cross_config: CrossBuildConfig) -> None:
    requirements = {
        'numpy>=2',
        "pywin32; sys_platform == 'win32'",
        "cffi>=1; platform_machine == 'aarch64'",
        "typing-extensions; python_version < '3.11'",
        "tomli; python_version >= '3.11' and os_name == 'posix'",
    }
    assert build._cross.filter_host_requirements(requirements, cross_config) == ['cffi>=1', 'numpy>=2', 'tomli']


def test_filter_host_requirements_invalid(cross_config: CrossBuildConfig) -> None:
    with pytest.raises(build.BuildException, match='Invalid host requirement'):
        build._cross.filter_host_requirements(['not a requirement!!'], cross_config)


def test_install_host_requirements_pip(mocker: pytest_mock.MockerFixture, cross_config: CrossBuildConfig) -> None:
    run_subprocess = mocker.patch('build._cross.run_subprocess')

    build._cross.install_host_requirements(['numpy>=2', 'cffi'], '/target', cross_config, installer='pip')

    (cmd,) = run_subprocess.call_args.args
    requirements_file = cmd[-1]
    assert cmd[:-2] == [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--target',
        '/target',
        '--only-binary=:all:',
        '--no-compile',
        '--no-input',
        '--python-version',
        '3.14',
        '--implementation',
        'cp',
        '--platform',
        'linux_aarch64',
        '--platform',
        'manylinux_2_17_aarch64',
        '--platform',
        'manylinux2014_aarch64',
        '--abi',
        'cp314',
        '--abi',
        'abi3',
        '--abi',
        'none',
    ]
    assert cmd[-2] == '-r'
    # the requirements file is removed after the installer ran
    assert not os.path.exists(requirements_file)


def test_install_host_requirements_pip_constraints(mocker: pytest_mock.MockerFixture, cross_config: CrossBuildConfig) -> None:
    run_subprocess = mocker.patch('build._cross.run_subprocess')

    build._cross.install_host_requirements(
        ['numpy'], '/target', cross_config, installer='pip', constraints_txt_path='constraints.txt'
    )

    (cmd,) = run_subprocess.call_args.args
    assert cmd[-2:] == ['-c', 'constraints.txt']


def test_install_host_requirements_uv(mocker: pytest_mock.MockerFixture, cross_config: CrossBuildConfig) -> None:
    run_subprocess = mocker.patch('build._cross.run_subprocess')
    mocker.patch('shutil.which', return_value='/usr/bin/uv')
    mocker.patch.dict(sys.modules, {'uv': None})

    build._cross.install_host_requirements(['numpy>=2'], '/target', cross_config, installer='uv')

    (cmd,) = run_subprocess.call_args.args
    assert cmd == [
        '/usr/bin/uv',
        'pip',
        'install',
        '--target',
        '/target',
        '--only-binary',
        ':all:',
        '--python-version',
        '3.14',
        '--python-platform',
        'aarch64-unknown-linux-gnu',
        '--python',
        sys.executable,
        'numpy>=2',
    ]


@pytest.mark.parametrize(
    ('platform', 'multiarch', 'glibc_version', 'expected'),
    [
        ('linux-aarch64', 'aarch64-linux-gnu', (2, 28), 'aarch64-manylinux_2_28'),
        ('linux-x86_64', 'x86_64-linux-musl', None, 'x86_64-unknown-linux-musl'),
        ('macosx-11.0-arm64', 'darwin', None, 'aarch64-apple-darwin'),
        ('macosx-10.9-x86_64', 'darwin', None, 'x86_64-apple-darwin'),
        ('win-amd64', None, None, 'x86_64-pc-windows-msvc'),
        ('win-arm64', None, None, 'aarch64-pc-windows-msvc'),
    ],
)
def test_uv_python_platform(
    platform: str, multiarch: str | None, glibc_version: tuple[int, int] | None, expected: str
) -> None:
    cross = CrossBuildConfig(
        build_details=make_build_details(platform, multiarch=multiarch), toolchain={}, glibc_version=glibc_version
    )
    assert build._cross._uv_python_platform(cross) == expected


def test_install_host_requirements_empty(mocker: pytest_mock.MockerFixture, cross_config: CrossBuildConfig) -> None:
    run_subprocess = mocker.patch('build._cross.run_subprocess')
    build._cross.install_host_requirements([], '/target', cross_config)
    run_subprocess.assert_not_called()


def test_constrain_to_build_environment() -> None:
    requirements = ['numpy>=2', 'pybind11', 'pyarrow<25,>=18', 'cffi']
    constrained, pinned = build._cross.constrain_to_build_environment(
        requirements, {'numpy': '2.4.6', 'pybind11': '3.1.0', 'unrelated': '1.0'}
    )
    assert constrained == ['numpy==2.4.6,>=2', 'pybind11==3.1.0', 'pyarrow<25,>=18', 'cffi']
    assert pinned == {'numpy': '2.4.6', 'pybind11': '3.1.0'}


def test_constrain_to_build_environment_canonical_names() -> None:
    constrained, pinned = build._cross.constrain_to_build_environment(['Ruamel.YAML'], {'ruamel-yaml': '0.18'})
    assert constrained == ['Ruamel.YAML==0.18']
    assert pinned == {'ruamel-yaml': '0.18'}


def test_build_environment_versions_current_interpreter() -> None:
    versions = build._cross.build_environment_versions(sys.executable, ['packaging', 'Not-Installed-Package'])
    assert versions == {'packaging': importlib.metadata.version('packaging')}


def test_build_environment_versions_other_interpreter(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch(
        'build._cross.subprocess.run',
        return_value=mocker.Mock(stdout=json.dumps(sys.path)),
    )
    versions = build._cross.build_environment_versions('/some/other/python', ['packaging'])
    assert versions == {'packaging': importlib.metadata.version('packaging')}


# -- CLI ----------------------------------------------------------------------


def test_cli_build_details(mocker: pytest_mock.MockerFixture, build_details_file: pathlib.Path) -> None:
    build_package = mocker.patch('build.__main__.build_package', autospec=True, return_value=['something'])
    mocker.patch.dict(os.environ, {'CC': 'aarch64-linux-gnu-gcc'})

    build.__main__.main(
        [
            '--wheel',
            '--build-details',
            os.fspath(build_details_file),
            '--toolchain',
            'STRIP=aarch64-linux-gnu-strip',
            '--host-glibc-version',
            '2.28',
        ]
    )

    cross = build_package.call_args.kwargs['cross']
    assert isinstance(cross, CrossBuildConfig)
    assert cross.build_details['platform'] == 'linux-aarch64'
    assert cross.toolchain['CC'] == 'aarch64-linux-gnu-gcc'
    assert cross.toolchain['STRIP'] == 'aarch64-linux-gnu-strip'
    assert cross.glibc_version == (2, 28)


def test_cli_no_build_details(mocker: pytest_mock.MockerFixture) -> None:
    build_package_via_sdist = mocker.patch('build.__main__.build_package_via_sdist', return_value=['something'])

    build.__main__.main([])

    assert 'cross' not in build_package_via_sdist.call_args.kwargs


@pytest.mark.parametrize(
    ('args', 'message'),
    [
        (['--toolchain', 'CC=gcc'], '--toolchain and --host-glibc-version: only allowed with --build-details'),
        (['--host-glibc-version', '2.28'], '--toolchain and --host-glibc-version: only allowed with --build-details'),
        (['--build-details', 'x.json', '--metadata'], '--build-details: not allowed with --metadata'),
    ],
)
def test_cli_build_details_conflicts(capsys: pytest.CaptureFixture[str], args: list[str], message: str) -> None:
    with pytest.raises(SystemExit):
        build.__main__.main(args)
    assert message in capsys.readouterr().err


def test_cli_build_details_missing_file(capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        build.__main__.main(['--wheel', '--build-details', os.fspath(tmp_path / 'missing.json')])
    assert 'build-details.json not found' in capsys.readouterr().err


# -- end to end ---------------------------------------------------------------


def test_cross_build_package(
    mocker: pytest_mock.MockerFixture,
    tmp_path: pathlib.Path,
    package_test_cross: str,
    cross_config: CrossBuildConfig,
) -> None:
    def fake_install(requirements: list[str], target: str, *_args: object, **_kwargs: object) -> None:
        for requirement in requirements:
            (pathlib.Path(target) / requirement.split('=')[0].strip()).mkdir()

    install = mocker.patch('build.__main__.install_host_requirements', side_effect=fake_install)
    outdir = tmp_path / 'dist'

    built = build.__main__.build_package(
        package_test_cross, outdir, ['wheel'], isolation=False, skip_dependency_check=True, cross=cross_config
    )

    assert built == ['test_cross-1.0.0-cp314-none-linux_aarch64.whl']
    requirements, target, cross = install.call_args.args
    assert cross is cross_config
    # pyproject.toml host-requires plus get_host_requires_for_cross_build_wheel, filtered for the host
    assert requirements == {'numpy', 'cffi', 'host-extra==1.0'}
    assert install.call_args.kwargs == {'installer': 'pip', 'constraints_txt_path': None}

    with zipfile.ZipFile(outdir / built[0]) as whl:
        info = json.loads(whl.read('cross.json'))
    assert info['build_details'] == cross_config.build_details
    assert info['host_requires_directory'] == target
    assert info['host_requires_contents'] == ['cffi', 'host-extra', 'numpy']
    assert info['toolchain'] == {'CC': 'aarch64-linux-gnu-gcc'}
    # the temporary host requirements directory is cleaned up
    assert not os.path.exists(target)


def test_cross_build_package_pins_host_requirements_to_build_environment(
    mocker: pytest_mock.MockerFixture,
    tmp_path: pathlib.Path,
    package_test_cross: str,
    cross_config: CrossBuildConfig,
) -> None:
    install = mocker.patch('build.__main__.install_host_requirements')
    # numpy is "installed" in the (non-isolated) build environment at 2.4.6
    mocker.patch('build.__main__.build_environment_versions', return_value={'numpy': '2.4.6'})

    build.__main__.build_package(
        package_test_cross, tmp_path / 'dist', ['wheel'], isolation=False, skip_dependency_check=True, cross=cross_config
    )

    requirements = install.call_args.args[0]
    assert requirements == {'numpy==2.4.6', 'cffi', 'host-extra==1.0'}


def test_cross_build_package_pin_failure_hint(
    mocker: pytest_mock.MockerFixture,
    tmp_path: pathlib.Path,
    package_test_cross: str,
    cross_config: CrossBuildConfig,
) -> None:
    error = subprocess.CalledProcessError(1, ['pip'])
    mocker.patch(
        'build.__main__.install_host_requirements',
        side_effect=build.FailedProcessError(error, 'Failed to install host requirements'),
    )
    mocker.patch('build.__main__.build_environment_versions', return_value={'numpy': '2.5.4'})

    with pytest.raises(build.BuildException, match='constrained to the versions installed in the build environment'):
        build.__main__.build_package(
            package_test_cross,
            tmp_path / 'dist',
            ['wheel'],
            isolation=False,
            skip_dependency_check=True,
            cross=cross_config,
        )


def test_cross_build_package_env_dir(
    mocker: pytest_mock.MockerFixture,
    tmp_path: pathlib.Path,
    package_test_cross: str,
    cross_config: CrossBuildConfig,
) -> None:
    mocker.patch('build.__main__.install_host_requirements')
    env_dir = tmp_path / 'env'
    env_dir.mkdir()
    # bypass the isolated environment but keep env_dir so the host requirements directory is placed there
    mocker.patch('build.__main__._bootstrap_build_env', side_effect=_fake_bootstrap(package_test_cross))

    build.__main__.build_package(
        package_test_cross, tmp_path / 'dist', ['wheel'], env_dir=os.fspath(env_dir), cross=cross_config
    )

    assert (env_dir / 'host-requires').is_dir()


def _fake_bootstrap(srcdir: str) -> Callable[..., AbstractContextManager[build.ProjectBuilder]]:
    @contextlib.contextmanager
    def bootstrap(*_args: object, **_kwargs: object) -> Generator[build.ProjectBuilder]:
        yield build.ProjectBuilder(srcdir)

    return bootstrap


def test_cross_build_package_sdist_is_not_cross_built(
    mocker: pytest_mock.MockerFixture, tmp_path: pathlib.Path, package_test_cross: str, cross_config: CrossBuildConfig
) -> None:
    install = mocker.patch('build.__main__.install_host_requirements')
    builder_build = mocker.patch('build.ProjectBuilder.build', return_value=os.fspath(tmp_path / 'test_cross-1.0.0.tar.gz'))

    build.__main__.build_package(
        package_test_cross, tmp_path, ['sdist'], isolation=False, skip_dependency_check=True, cross=cross_config
    )

    builder_build.assert_called_once()
    install.assert_not_called()


def test_cross_build_package_editable(tmp_path: pathlib.Path, package_test_cross: str, cross_config: CrossBuildConfig) -> None:
    with pytest.raises(build.BuildException, match='Cannot cross build editable'):
        build.__main__.build_package(
            package_test_cross, tmp_path, ['editable'], isolation=False, skip_dependency_check=True, cross=cross_config
        )


def test_cross_build_unsupported_backend(
    tmp_path: pathlib.Path, package_test_no_prepare: str, cross_config: CrossBuildConfig
) -> None:
    with pytest.raises(build.BuildException, match='does not support cross builds'):
        build.__main__.build_package(
            package_test_no_prepare, tmp_path, ['wheel'], isolation=False, skip_dependency_check=True, cross=cross_config
        )


def test_project_builder_cross_build_hook_missing(
    tmp_path: pathlib.Path, package_test_no_prepare: str, cross_config: CrossBuildConfig
) -> None:
    builder = build.ProjectBuilder(package_test_no_prepare)
    assert not builder.supports_cross_build()
    with pytest.raises(build.BuildBackendException, match='does not support cross builds'):
        builder.cross_build(tmp_path, build_details=cross_config.build_details)


def test_project_builder_get_build_requires(package_test_cross: str, package_test_no_prepare: str) -> None:
    assert build.ProjectBuilder(package_test_cross).get_build_requires_for_cross_build({}, make_build_details()) == {
        'build-extra == 1.0'
    }
    # backends that do not define the hook contribute nothing (no fallback to get_requires_for_build_wheel)
    assert build.ProjectBuilder(package_test_no_prepare).get_build_requires_for_cross_build({}, make_build_details()) == set()


def test_cross_build_isolated_installs_cross_build_requires(
    mocker: pytest_mock.MockerFixture, package_test_cross: str, cross_config: CrossBuildConfig
) -> None:
    cross_build = mocker.patch('build.ProjectBuilder.cross_build', return_value='something.whl')
    mocker.patch('build.ProjectBuilder.supports_cross_build', return_value=True)
    mocker.patch('build.ProjectBuilder.get_host_requires_for_cross_build', return_value=set())
    get_requires = mocker.patch('build.ProjectBuilder.get_requires_for_build')
    get_build_requires = mocker.patch('build.ProjectBuilder.get_build_requires_for_cross_build', return_value={'build-extra'})
    install = mocker.patch('build.env.DefaultIsolatedEnv.install')
    mocker.patch('build.__main__.install_host_requirements')
    mocker.patch('build.__main__._log_dependency_versions')

    build.__main__.build_package(package_test_cross, '.', ['wheel'], cross=cross_config)

    get_build_requires.assert_called_once_with(None, cross_config.build_details)
    get_requires.assert_not_called()
    install.assert_any_call({'build-extra'}, constraints_txt_path=None)
    cross_build.assert_called_once()


def test_cross_build_no_isolation_checks_cross_build_requires(
    mocker: pytest_mock.MockerFixture, package_test_cross: str, cross_config: CrossBuildConfig
) -> None:
    mocker.patch('build.__main__.install_host_requirements')
    error = mocker.patch('build.__main__._error', side_effect=SystemExit)

    with pytest.raises(SystemExit):
        build.__main__.build_package(package_test_cross, '.', ['wheel'], isolation=False, cross=cross_config)

    # the test backend's get_build_requires_for_cross_build_wheel requires a package that is not installed
    assert 'build-extra' in error.call_args.args[0]


def test_project_builder_get_host_requires(package_test_cross: str, package_test_no_prepare: str) -> None:
    assert build.ProjectBuilder(package_test_cross).get_host_requires_for_cross_build({}, make_build_details()) == {
        'host-extra == 1.0'
    }
    # backends that do not define the hook contribute nothing
    assert build.ProjectBuilder(package_test_no_prepare).get_host_requires_for_cross_build({}, make_build_details()) == set()
