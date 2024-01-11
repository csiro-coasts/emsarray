import re
from collections.abc import Iterable
from typing import Callable, cast

import yaml
from docutils import nodes, utils
from docutils.parsers.rst import Directive, directives, roles
from docutils.parsers.rst.states import Inliner
from sphinx.application import Sphinx

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
    ) -> tuple[list, list]:
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


class Citation(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'citation_file': directives.path,
        'format': lambda a: directives.choice(a, [
            'apa', 'biblatex',
        ])
    }

    def load_citation_file(self) -> dict:
        citation_file = self.options['citation_file']
        with open(citation_file, 'r') as f:
            return cast(dict, yaml.load(f, yaml.Loader))

    def run(self) -> list[nodes.Node]:
        if self.options['format'] == 'apa':
            return self.run_apa()
        elif self.options['format'] == 'biblatex':
            return self.run_biblatex()
        else:
            raise ValueError("Unknown format")

    def run_apa(self) -> list[nodes.Node]:
        citation = self.load_citation_file()
        names = self.comma_ampersand_join(map(self.apa_name, citation['authors']))
        year = citation['date-released'].year
        title = citation['title']
        version = citation['version']
        publisher = self.publisher(citation)
        url = citation['url']

        return [nodes.block_quote('', nodes.paragraph(
            '', '',
            nodes.Text(f'{names}. ({year}). '),
            nodes.emphasis(text=title),
            nodes.Text(f' (Version {version}) [Software]. {publisher}. {url}'),
        ))]

    def apa_name(self, entity: dict) -> str:
        if 'name' in entity:
            return str(entity['name'])
        return '{}, {}.'.format(entity['family-names'], entity['given-names'][0])

    def comma_ampersand_join(self, items: Iterable[str]) -> str:
        items = list(items)
        if len(items) == 1:
            return items[0]
        return '{}, & {}'.format(', '.join(items[:-1]), items[-1])

    def run_biblatex(self) -> list[nodes.Node]:
        citation = self.load_citation_file()

        year = citation['date-released'].year

        attributes = {
            'title': citation['title'],
            'author': ' and '.join(map(self.biblatex_name, citation['authors'])),
            'year': year,
            'type': 'software',
            'url': citation['url'],
            'version': citation['version'],
            'publisher': self.publisher(citation),
        }
        ref = f'emsarray_{year}'
        lines = '\n'.join(
            f'    {attribute} = {{{value}}},'
            for attribute, value in attributes.items()
        )
        biblatex_entry = f'@software{{{ref},\n{lines}\n}}'

        return [nodes.literal_block(biblatex_entry, biblatex_entry)]

    def biblatex_name(self, author: dict) -> str:
        if 'name' in author:
            return '{' + str(author['name']) + '}'
        return '{}, {}'.format(author['family-names'], author['given-names'])

    def publisher(self, citation: dict) -> str:
        return next(
            str(ref['publisher']['name'])
            for ref in citation['references']
            if ref['type'] == 'generic' and 'publisher' in ref)


def setup(app: Sphinx) -> dict:
    app.add_role('pr', github_pr)
    app.add_role('issue', github_issue)
    app.add_directive('citation', Citation)

    return {'parallel_read_safe': True}
