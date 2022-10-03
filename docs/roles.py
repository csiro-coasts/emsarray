import re
from typing import Any, Callable, Tuple

from docutils import nodes, utils
from docutils.parsers.rst import roles
from docutils.parsers.rst.states import Inliner

_GITHUB_NAME = r'[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*'
GITHUB_FULL_REF = re.compile(
    r'^'
    r'(?P<repo>' + _GITHUB_NAME + '/' + _GITHUB_NAME + ')'
    r'#(?P<num>[1-9][\d]*)'
    r'$'
)

GITHUB_DEFAULT_REPO = 'csiro-coasts/emsarray'


def _github_link(
    prefix: str,
    url_part: str,
) -> Callable:
    def role_fn(
        role: str,
        rawtext: str,
        text: str,
        lineno: int,
        inliner: Inliner,
        options: dict = {},
        content: list = [],
    ) -> Tuple[list, list]:
        match = GITHUB_FULL_REF.match(utils.unescape(text))
        if match is not None:
            repo = match.group('repo')
            num = int(match.group('num'))
        else:
            try:
                repo = GITHUB_DEFAULT_REPO
                num = int(utils.unescape(text))
                if num <= 0:
                    raise ValueError
            except ValueError:
                msg = inliner.reporter.error(
                    'Github issue reference must be a positive integer, '
                    'or a full user/repo#num reference',
                    line=lineno)
                prb = inliner.problematic(rawtext, rawtext, msg)
                return [prb], [msg]

        refuri = f'https://github.com/{repo}/{url_part}/{num}'
        if repo == GITHUB_DEFAULT_REPO:
            display = f'{prefix} #{num}'
        else:
            display = f'{prefix} {repo}#{num}'

        roles.set_classes(options)
        ref = nodes.reference(rawtext, display, refuri=refuri, **options)

        return [ref], []
    return role_fn


github_pr = _github_link('pull request', 'pull')
github_issue = _github_link('issue', 'issues')


def setup(arg: Any) -> None:
    roles.register_canonical_role('pr', github_pr)
    roles.register_canonical_role('issue', github_issue)
    return {'parallel_read_safe': True}
