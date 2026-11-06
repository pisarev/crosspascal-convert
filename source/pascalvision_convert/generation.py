# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
import math
import secrets
import numpy as np


class Generation:
    def __init__(self, options, prompt_ids):
        options = dict(options or {})
        allowed = {'sampling', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'seed', 'effort'}
        if options.keys() - allowed:
            raise ValueError('Unknown generation setting: ' + ', '.join(sorted(options.keys() - allowed)))
        self.mode = options.get('sampling', 'greedy')
        self.temperature = options.get('temperature', 0.7)
        self.top_p = options.get('top_p', 0.8)
        self.top_k = options.get('top_k', 20)
        self.penalty = options.get('repetition_penalty', 1.1)
        self.seed = options.get('seed', -1)
        self.effort = options.get('effort', 'full_context')
        if self.mode not in ('greedy', 'sample'):
            raise ValueError('Select greedy or sample decoding')
        for name, value, low, high in [('temperature', self.temperature, 0.01, 2),
                                      ('top_p', self.top_p, 1e-12, 1),
                                      ('repetition_penalty', self.penalty, 1, 2)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Invalid ' + name)
        if type(self.top_k) is not int or not 1 <= self.top_k <= 128:
            raise ValueError('Top-K must be between 1 and 128')
        if type(self.seed) is not int or not -1 <= self.seed <= 4294967295:
            raise ValueError('Seed must be -1 (random) or an unsigned 32-bit integer')
        if self.effort not in ('full_context', 'brief', 'balanced', 'detailed'):
            raise ValueError('Unknown answer budget')
        if self.seed < 0:
            self.seed = secrets.randbits(32)
        self.rng = np.random.default_rng(self.seed)
        self.seen = set(prompt_ids)

    def limit(self, manual, available):
        if type(manual) is not int or manual < 0:
            raise ValueError('Use 0 for automatic output or a positive manual limit')
        if manual and manual < available:
            return manual, 'manual_limit'
        budget = {'full_context': 0, 'brief': 128, 'balanced': 512, 'detailed': 1024}[self.effort]
        if not manual and budget and budget < available:
            return budget, 'effort_limit'
        return available, 'context_limit'

    def choose(self, logits):
        values = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
        if not np.isfinite(values).all():
            raise RuntimeError('Non-finite model logits')
        if self.mode == 'greedy':
            return int(values.argmax())
        ids = np.asarray(sorted(i for i in self.seen if 0 <= i < len(values)), dtype=np.int64)
        if len(ids):
            values[ids] = np.where(values[ids] < 0, values[ids] * self.penalty, values[ids] / self.penalty)
        count = min(self.top_k, len(values))
        candidates = np.argsort(-values, kind='stable')[:count]
        weights = np.exp((values[candidates] - values[candidates[0]]) / self.temperature)
        weights /= weights.sum()
        kept = min(count, int(np.searchsorted(np.cumsum(weights), self.top_p)) + 1)
        weights = weights[:kept]
        token = int(self.rng.choice(candidates[:kept], p=weights / weights.sum()))
        self.seen.add(token)
        return token

    def report(self):
        return dict(sampling=self.mode, temperature=self.temperature, top_p=self.top_p,
                    top_k=self.top_k, repetition_penalty=self.penalty, seed=self.seed,
                    effort=self.effort)
