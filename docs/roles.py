from typing import Any, Callable, Tuple

from docutils import nodes, utils
from docutils.parsers.rst import roles
from docutils.parsers.rst.states import Inliner


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
        try:
            num = int(utils.unescape(text))
            if num <= 0:
                raise ValueError
        except ValueError:
            msg = inliner.reporter.error(
                f'PR number must be a positive integer, not {text!r}',
                line=lineno)
            prb = inliner.problematic(rawtext, rawtext, msg)
            return [prb], [msg]

        # Base URL mainly used by inliner.pep_reference; so this is correct:
        repo = 'https://github.com/csiro-coasts/emsarray'
        refuri = f'{repo}/{url_part}/{num}'

        roles.set_classes(options)
        ref = nodes.reference(rawtext, f'{prefix} #{text}', refuri=refuri, **options)

        return [ref], []
    return role_fn


github_pr = _github_link('pull request', 'pull')
github_issue = _github_link('issue', 'issues')


def setup(arg: Any) -> None:
    roles.register_canonical_role('pr', github_pr)
    roles.register_canonical_role('issue', github_issue)
    return {'parallel_read_safe': True}
