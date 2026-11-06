# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Conversion reports for the CLI and wizard. Each step can be done, refused or unchecked."""

DONE = 'done'
REFUSED = 'refused'
UNCHECKED = 'unchecked'

STATE_NAMES = {
    DONE: 'done',
    REFUSED: 'refused',
    UNCHECKED: 'not checked',
}


class Step(object):
    """One step of the work: name, state, detail lines and numbers."""

    def __init__(self, name, state=UNCHECKED, lines=None, numbers=None):
        self.name = name
        self.state = state
        self.lines = list(lines or [])
        self.numbers = dict(numbers or {})

    def say(self, line):
        self.lines.append(line)
        return self

    def done(self, line=None):
        self.state = DONE
        return self.say(line) if line else self

    def refuse(self, line=None):
        self.state = REFUSED
        return self.say(line) if line else self

    def as_dict(self):
        return {
            'name': self.name,
            'state': self.state,
            'state_name': STATE_NAMES[self.state],
            'lines': list(self.lines),
            'numbers': dict(self.numbers),
        }


class Report(object):
    """A sequence of steps. Sound as long as none of them refused."""

    def __init__(self, title=''):
        self.title = title
        self.steps = []
        self.summary = {}

    def step(self, name):
        fresh = Step(name)
        self.steps.append(fresh)
        return fresh

    @property
    def sound(self):
        return not any(s.state == REFUSED for s in self.steps)

    @property
    def unchecked(self):
        return [s.name for s in self.steps if s.state == UNCHECKED]

    def as_dict(self):
        return {
            'title': self.title,
            'sound': self.sound,
            'unchecked': self.unchecked,
            'steps': [s.as_dict() for s in self.steps],
            'summary': dict(self.summary),
        }

    def as_text(self):
        """Flat form for a build log. Exactly what the wizard shows."""
        out = []
        if self.title:
            out.append(self.title)
            out.append('')
        for step in self.steps:
            out.append('[%s] %s' % (STATE_NAMES[step.state], step.name))
            for line in step.lines:
                out.append('    ' + line)
        if self.unchecked:
            out.append('')
            out.append('NOT CHECKED: ' + ', '.join(self.unchecked))
        out.append('')
        verification = self.summary.get('verification_status')
        if verification:
            out.append('VERIFICATION: ' + verification)
            result = ('REFUSED' if not self.sound else
                      'VERIFIED' if verification == 'VERIFIED_LOCAL' else
                      'COMPILED, NOT VERIFIED')
        else:
            result = 'sound' if self.sound else 'REFUSED'
        out.append('RESULT: ' + result)
        return chr(10).join(out)
