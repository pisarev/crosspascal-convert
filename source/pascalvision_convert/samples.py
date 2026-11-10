# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


import os

from . import pipeline
from .report import DONE, Report

SAMPLES_VARIABLE = 'PASCALVISION_SAMPLES'
DEFAULT_DIRECTORY = os.path.abspath('samples')


def samples_directory(given=None):
    if given:
        return given
    from_environment = os.environ.get(SAMPLES_VARIABLE, '')
    if from_environment:
        return from_environment
    return DEFAULT_DIRECTORY

SAMPLES = (
    {
        'name': 'yolo11n',
        'file': 'yolo11n.simplified.onnx',
        'kind': 'onnx',
        'what': 'detector, input 1x3x640x640, output 1x84x8400',
        'recorded': {'smooth': 1.796e-06, 'noise': 2.176e-06, 'flat': 3.221e-06},
        'measured': '31.08.2026, clean environment, apache-tvm 0.26.0',
    },
    {
        'name': 'mobilenet_v2',
        'file': 'mobilenet_v2_1.0_224.tflite',
        'kind': 'tflite',
        'what': 'classifier, input 1x224x224x3 NHWC, output 1x1001',
        'recorded': {'smooth': 4.321e-07, 'noise': 9.239e-07, 'flat': 3.651e-07},
        'measured': '31.08.2026, reference is the ai-edge-litert interpreter',
    },
)

NOTEWORTHY_RATIO = 100


def check_installation(directory=None, work_dir=None, on_step=None):
    directory = samples_directory(directory)
    work_dir = work_dir or os.path.join(directory, 'installation-check')
    report = Report('Installation check on known-good models')

    for sample in SAMPLES:
        step = report.step('Sample %s' % sample['name'])
        step.say(sample['what'])
        path = os.path.join(directory, sample['file'])
        if not os.path.isfile(path):
            step.say('no such file: %s' % path)
            step.say('this is NOT CHECKED - neither a pass nor a failure')
            if on_step:
                on_step(step)
            continue

        run = pipeline.run(path, os.path.join(work_dir, sample['name']), 'cpu')
        comparison = None
        for other in run.steps:
            if other.name.startswith('Comparison'):
                comparison = other
        if comparison is None:
            step.refuse('the comparison was never reached; see the build report')
            for other in run.steps:
                for line in other.lines:
                    step.say(line)
            if on_step:
                on_step(step)
            continue

        for label, recorded in sample['recorded'].items():
            keys = [k for k in comparison.numbers if k.startswith(label + '.')]
            if not keys:
                step.say('%-8s recorded %.3e, no fresh number' % (label, recorded))
                continue
            fresh = max(comparison.numbers[k] for k in keys)
            note = ''
            if fresh > recorded * NOTEWORTHY_RATIO:
                note = '   ABOVE THE RECORDED VALUE by more than %dx' % NOTEWORTHY_RATIO
            step.say('%-8s recorded %.3e, now %.3e%s' % (label, recorded, fresh, note))
        step.say('measured: %s' % sample['measured'])
        if not run.sound:
            step.refuse('comparison failed')
        elif comparison.state == DONE:
            step.done('comparison passed')
        else:
            step.say('comparison not checked')
            for line in comparison.lines:
                step.say(line)
        if on_step:
            on_step(step)

    return report
