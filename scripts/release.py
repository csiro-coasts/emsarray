import argparse
import configparser
import datetime
import pathlib
import re
import shlex
import subprocess
import sys
from typing import List, Optional

PROJECT = pathlib.Path(__file__).parent.parent

# These files contain the version string that needs to be updated
citation_file = pathlib.Path('CITATION.cff')
version_files = [
    pathlib.Path('setup.cfg'),
    citation_file,
]

release_notes_index = pathlib.Path('docs/releases/index.rst')
release_notes_dev = pathlib.Path('docs/releases/development.rst')
release_notes_dev_header = """
=============================
Next release (in development)
=============================

* ...
""".lstrip()


def main(
    args: Optional[List[str]] = None,
) -> None:
    parser = argparse.ArgumentParser()
    add_options(parser)

    try:
        opts = parser.parse_args(args)
        opts.func(opts)

    except CommandError as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)

    except KeyboardInterrupt:
        sys.exit(1)


def add_options(parser: argparse.ArgumentParser) -> None:
    parser.description = __doc__
    subparsers = parser.add_subparsers(title='commands', required=True)

    add_pre_options(subparsers.add_parser('pre'))
    add_tag_options(subparsers.add_parser('tag'))
    add_post_options(subparsers.add_parser('post'))


def add_pre_options(parser: argparse.ArgumentParser) -> None:
    parser.description = "Get the project ready for making a release"
    parser.set_defaults(func=pre_release, old_version='placeholder')
    parser.add_argument(
        'new_version', type=str,
        help=("The new version to bump to"))
    parser.add_argument(
        '--remote', type=str, default='origin',
        help=("The name of the csiro-coasts/emsarray remote. Defaults to 'origin'"))
    parser.add_argument(
        '--ref', type=str, default='origin/main',
        help=("The ref to branch from. Defaults to 'origin/main'"))
    parser.add_argument(
        '--release-date', type=argparse_date, default=datetime.date.today(),
        help=('The release date in YYYY-MM-DD format. Defaults to today'))


def add_tag_options(parser: argparse.ArgumentParser) -> None:
    parser.description = "Tag the current commit for release"
    parser.set_defaults(func=tag_release, version='placeholder')
    parser.add_argument(
        '--remote', type=str, default='origin',
        help=("The name of the csiro-coasts/emsarray remote. Defaults to 'origin'"))
    parser.add_argument(
        '--ref', type=str, default='origin/main',
        help=("The ref to tag. Defaults to 'origin/main'"))


def add_post_options(parser: argparse.ArgumentParser) -> None:
    parser.description = "Get the project ready for development after a release"
    parser.set_defaults(func=post_release, version='placeholder')
    parser.add_argument(
        '--remote', type=str, default='origin',
        help=("The name of the csiro-coasts/emsarray remote. Defaults to 'origin'"))
    parser.add_argument(
        '--ref', type=str, default='origin/main',
        help=("The ref to branch from. Defaults to 'origin/main'"))


def pre_release(opts: argparse.Namespace) -> None:
    old_version = get_current_version()
    print(f"Bumping from {old_version} to {opts.new_version}")

    git_check(opts.remote)

    branch_name = f'release-{opts.new_version}'
    call('git', 'checkout', '-b', branch_name, opts.ref)

    release_date_re = re.compile(r'date-released: \d+-\d+-\d+')
    new_release_date = f'date-released: {opts.release_date.isoformat()}'
    replace_in_file(citation_file, release_date_re, new_release_date)

    old_version_pattern = r'(?<!\.)' + re.escape(old_version) + r'(?!\.)'
    old_version_re = re.compile(old_version_pattern)
    for path in version_files:
        print("Updating version in", path)
        replace_in_file(path, old_version_re, opts.new_version)
        call('git', 'add', str(path))

    new_release_notes = release_notes_dev.parent / f'{opts.new_version}.rst'
    print(f"Renaming {release_notes_dev} to {new_release_notes}...")
    call('git', 'mv', str(release_notes_dev), str(new_release_notes))

    heading_re = re.compile('.*?^=+$.*?^=+$', re.MULTILINE | re.DOTALL)
    version = opts.new_version
    ruler = '=' * len(version)
    new_heading = '\n'.join([
        ruler,
        version,
        ruler,
        '',
        f'Released on {opts.release_date.isoformat()}',
    ])
    replace_in_file(new_release_notes, heading_re, new_heading)
    call('git', 'add', str(new_release_notes))

    replace_in_file(
        release_notes_index,
        re.compile(fr'^( *){release_notes_dev.stem}$', re.MULTILINE),
        r'\g<1>' + new_release_notes.stem)
    call('git', 'add', str(release_notes_index))

    git_commit(f'Bump version to v{opts.new_version}')
    maybe_push(opts.remote, branch_name)


def tag_release(opts: argparse.Namespace) -> None:
    version = get_current_version()
    print(f"Tagging release {version}")

    git_check(opts.remote)

    call('git', 'show', opts.ref)
    if not yn("Is this the correct commit to tag?"):
        raise CommandError("Aborted")
    tag_name = f'v{version}'
    call(
        'git', 'tag', tag_name, opts.ref,
        '-a', '-m', f'Version {version}')

    maybe_push(opts.remote, tag_name)


def post_release(opts: argparse.Namespace) -> None:
    version = get_current_version()
    print("Preparing for development")

    git_check(opts.remote)

    branch_name = f'post-{version}-release'
    call('git', 'checkout', '-b', branch_name, opts.ref)

    release_notes_dev.write_text(release_notes_dev_header)
    toctree_re = re.compile(r'^\.\. toctree::$\n(^ +.*$\n)*^$\n', re.MULTILINE)
    replace_in_file(
        release_notes_index,
        toctree_re,
        r'\g<0>   ' + release_notes_dev.stem + '\n')
    call('git', 'add', str(release_notes_dev), str(release_notes_index))

    git_commit(f'Prepare for development after version {opts.version} release')
    maybe_push(opts.remote, branch_name)


def git_check(remote: str) -> None:
    status = output('git', 'status', '--porcelain', '--untracked-files=no')
    if status != b'':
        raise CommandError(f'Working tree is dirty, aborting!\n{status.decode()}')
    call('git', 'fetch', remote)


def git_commit(message: str, *args: str) -> None:
    call('git', '--no-pager', 'diff', '--cached')
    call('git', 'commit', '-m', message, *args)


def maybe_push(
    remote: str,
    ref_name: str,
) -> None:
    if not yn(f"Push {ref_name!r} to {remote!r}?"):
        raise CommandError("Aborted")

    call('git', 'push', remote, ref_name)


def get_current_version() -> str:
    parser = configparser.ConfigParser()
    with open(PROJECT / 'setup.cfg') as f:
        parser.read_file(f)
    return parser.get('metadata', 'version')


def call(*args: str) -> None:
    print('$', shlex.join(args))
    try:
        subprocess.check_call(args, cwd=PROJECT)
    except subprocess.CalledProcessError as err:
        raise CommandError("Aborted") from err


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


def argparse_date(value: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'not a valid date: {value!r}')


class CommandError(Exception):
    pass


if __name__ == '__main__':
    main()
