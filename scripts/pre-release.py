#!/usr/bin/env python
"""
Get the project ready for making a release
"""
import argparse
import configparser
import pathlib
import re
import shlex
import subprocess
import sys
from typing import List, Optional

PROJECT = pathlib.Path(__file__).parent.parent

version_files = [
    pathlib.Path('setup.cfg'),
]

release_notes_index = pathlib.Path('docs/releases/index.rst')
release_notes_dev = pathlib.Path('docs/releases/development.rst')


def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    add_options(parser)
    opts = parser.parse_args(args)

    opts.old_version = get_current_version()

    print(f"Bumping from {opts.old_version} to {opts.new_version}")

    try:
        git_check(opts)
        branch_name = git_branch(opts)
        bump_version(opts)
        finalise_release_notes(opts)
        git_commit(opts)
        maybe_push(opts, branch_name)

    except CommandError as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)

    except KeyboardInterrupt:
        sys.exit(1)


def add_options(parser: argparse.ArgumentParser) -> None:
    parser.description = __doc__
    parser.set_defaults(old_version='placeholder')
    parser.add_argument(
        'new_version', type=str,
        help=("The new version to bump to"))
    parser.add_argument(
        '--remote', type=str, default='origin',
        help=("The name of the csiro-coasts/emsarray remote. Defaults to 'origin'"))


def git_check(opts: argparse.Namespace) -> None:
    status = output('git', 'status', '--porcelain', '--untracked-files=no')
    if status != b'':
        raise CommandError(f'Working tree is dirty, aborting!\n{status.decode()}')
    call('git', 'fetch', opts.remote)


def git_branch(opts: argparse.Namespace) -> str:
    branch_name = f'release/{opts.new_version}'
    call('git', 'checkout', '-b', branch_name, f'{opts.remote}/main')
    return branch_name


def bump_version(opts: argparse.Namespace) -> None:
    old_version_pattern = r'(?<!\.)' + re.escape(opts.old_version) + r'(?!\.)'
    old_version_re = re.compile(old_version_pattern)
    for path in version_files:
        print("Updating version in", path)
        replace_in_file(path, old_version_re, opts.new_version)
        call('git', 'add', str(path))


def finalise_release_notes(opts: argparse.Namespace) -> None:
    new_release_notes = release_notes_dev.parent / f'{opts.new_version}.rst'
    print(f"Renaming {release_notes_dev} to {new_release_notes}...")
    call('git', 'mv', str(release_notes_dev), str(new_release_notes))

    heading_re = re.compile('.*?^=+$.*?^=+$', re.MULTILINE | re.DOTALL)
    new_heading = '{ruler}\n{version}\n{ruler}'.format(
        version=opts.new_version, ruler='=' * len(opts.new_version))
    replace_in_file(new_release_notes, heading_re, new_heading)
    call('git', 'add', str(new_release_notes))

    replace_in_file(
        release_notes_index,
        re.compile(fr'^( *){release_notes_dev.stem}$', re.MULTILINE),
        r'\g<1>' + new_release_notes.stem)
    call('git', 'add', str(release_notes_index))


def git_commit(opts: argparse.Namespace) -> None:
    call('git', '--no-pager', 'diff', '--cached')
    call('git', 'commit', '-m', f'Bump version to v{opts.new_version}')


def maybe_push(
    opts: argparse.Namespace,
    branch_name: str,
) -> None:
    if not yn(f"Push branch {branch_name!r} to {opts.remote}?"):
        raise CommandError("Aborted")

    call('git', 'push', opts.remote, branch_name)


def get_current_version() -> str:
    parser = configparser.ConfigParser()
    with open(PROJECT / 'setup.cfg') as f:
        parser.read_file(f)
    return parser.get('metadata', 'version')


def call(*args: str) -> None:
    print('$', shlex.join(args))
    subprocess.check_call(args, cwd=PROJECT)


def output(*args: str) -> bytes:
    return subprocess.check_output(args, cwd=PROJECT)


def yn(
    prompt: str,
    default: Optional[bool] = None,
) -> bool:
    examples = {True: '[Yn]', False: '[yN]', None: '[yn]'}[default]
    prompt = f'{prompt.strip()} {examples} '

    while True:
        response = input(prompt).lower()
        if response in {'y', 'yes'}:
            return True
        elif response in {'n', 'no'}:
            return False
        elif response == '' and default is not None:
            return default
        else:
            print("Invalid response '{response!r}'. Please enter y or n.")


def replace_in_file(
    path: pathlib.Path,
    pattern: re.Pattern,
    replacement: str,
) -> None:
    full_path = PROJECT / path
    old_content = full_path.read_text()
    if pattern.search(old_content) is None:
        raise CommandError(f"Error updating {path}: pattern not found!")
    new_content = pattern.sub(replacement, old_content)
    full_path.write_text(new_content)


class CommandError(Exception):
    pass


if __name__ == '__main__':
    main()
