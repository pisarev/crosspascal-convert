# -*- coding: utf-8 -*-
# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Write and read the build recipe - the INPUTS of a build, so CI can repeat it.

WHY A RECIPE FILE, AND WHY IT IS NOT THE MANIFEST. Two files leave a build, and
they answer two different questions. The manifest (manifest.txt / manifest.json)
is the runtime CONTRACT of the finished model: what the facade must feed it and
how to read what comes back. The recipe (recipe.json) is the INPUT of the build:
the model source, the target, the shape, the tuning choice and the model contract
- everything a second machine needs to produce that same artifact with no human
in the loop. The manifest describes a model that passed; the recipe describes how
to make one. Merging them would tie the passport of a verified model to the wish
that built it, and the master design keeps the two apart on purpose
(state/MASTER-DESIGN.md, point 4: the recipe is NOT to be merged with the manifest).

WHY THE RECIPE IS WRITTEN ON EVERY BUILD, THE MANIFEST ONLY SOMETIMES. The
manifest is a passport and is issued only when the recipe named a task to
discriminate on; without a task there is nothing to write. The recipe is the
wish, and the wish always exists - so it is written whenever a build is carried
through, and the build can always be repeated.

Only the standard library is used (json). The recipe is plain dict / list / str /
int / float / bool / None, so it maps straight onto JSON and back.
"""

import json
import os
import re

from .contract import build_contract_input
from . import onnx_closure, source_freeze, paddle_closure
from .build_plan import validate_recorded as validate_build_plan
from .targets import (tvm_target_spec, validate_target_sm,
                      validate_recorded_profile)





RECIPE_VERSION = 1
RECIPE_PROFILE_VERSION = 2
RECIPE_FAMILY_LICENSE_CLAIM_VERSION = 3


def _checked_declared_model_license(value):
    """A caller label is input data, never an attested model licence.

    Keep the recipe bound to the same single-line length/control rules as the
    Pascal manifest emitter; otherwise a native LLM recipe could preserve a
    misleading multi-record label without ever visiting that emitter.
    """
    if value is None:
        return None
    if (type(value) is not str or not value.strip() or len(value) > 1024 or
            any(ord(char) < 32 or ord(char) == 127 or
                char in '\u0085\u2028\u2029' for char in value)):
        raise ValueError('model license must be a bounded single-line declaration')
    return value


def _checked_model_license_claim(claim, value):
    """Validate a caller-only licence marker, never legal clearance."""
    if (not isinstance(claim, dict) or set(claim) != {
            'schema', 'version', 'license', 'basis',
            'redistribution_clearance'} or
            claim['schema'] != 'crosspascal.model-license-claim' or
            type(claim['version']) is not int or claim['version'] != 1 or
            claim['license'] != value or
            claim['basis'] != 'CALLER_DECLARED_NOT_ATTESTED' or
            claim['redistribution_clearance'] != 'NOT_DETERMINED' or
            value is None):
        raise ValueError('model license claim missing/forged/unverified status')
    return claim


def validate_family_license_receipt(recipe):
    """Preserve legacy recipes; require the caller-only marker in new family v3.

    This is schema completeness, not model-license or redistribution attestation.
    An unsigned legacy-version downgrade is not authenticated by this check.
    """
    value = _checked_declared_model_license(recipe.get('license'))
    version = recipe.get('recipe_version')
    if type(version) is not int:
        raise ValueError('invalid recipe version for model license claim')
    claim_present = 'model_license_claim' in recipe
    if version == RECIPE_FAMILY_LICENSE_CLAIM_VERSION:
        family = recipe.get('family_builder')
        plan = recipe.get('build_plan')
        if (not isinstance(family, dict) or family.get('adapter') != 'llm' or
                not isinstance(plan, dict) or
                plan.get('route') != 'family_full_build' or
                value is None or not claim_present):
            raise ValueError('native family v3 requires model license claim')
    if claim_present:
        if recipe.get('family_builder') is None:
            raise ValueError('model license claim requires native family recipe')
        _checked_model_license_claim(recipe['model_license_claim'], value)
    return value


def validate_paddle_selected_source(closure, *, source=None):
    """Check exact versioned Paddle scan receipt, without trusting its SHA label.

    This structural check is safe for detached recipe inspection. Callers must
    independently rescan original model AND dictionary before child execution.
    """
    keys = {'schema', 'schema_version', 'source_root', 'role',
            'graph_file', 'entries', 'file_count', 'total_bytes',
            'dictionary', 'tree_sha256', 'status', 'verification_status',
            'release_grade_immutable'}
    if not isinstance(closure, dict) or set(closure) != keys:
        raise ValueError('Paddle selected source receipt keys missing/unknown')
    if (closure['schema'] != paddle_closure.SCHEMA or
            type(closure['schema_version']) is not int or
            closure['schema_version'] != paddle_closure.SCHEMA_VERSION or
            closure['status'] != paddle_closure.STATUS or
            closure['verification_status'] != 'NOT_CHECKED' or
            closure['release_grade_immutable'] is not False or
            closure['role'] not in ('det', 'rec') or
            closure['graph_file'] not in ('inference.json', 'inference.pdmodel')):
        raise ValueError('Paddle selected source schema/status/role invalid')
    root = closure['source_root']
    if (type(root) is not str or
            str(paddle_closure._raw_absolute(root, 'Paddle receipt root')) != root or
            (source is not None and root != source)):
        raise ValueError('Paddle selected source identity differs from recipe')
    dictionary = closure['dictionary']
    if closure['role'] == 'det':
        if dictionary is not None:
            raise ValueError('Paddle det receipt cannot include dictionary')
    else:
        if not isinstance(dictionary, dict) or set(dictionary) != {
                'reference', 'path', 'bytes', 'sha256'}:
            raise ValueError('Paddle rec receipt requires exact dictionary')
        reference = dictionary['reference']
        if (type(reference) is not str or
                str(paddle_closure._raw_absolute(
                    reference, 'Paddle selected dictionary')) != reference or
                dictionary['path'] !=
                'dictionary/' + os.path.basename(reference) or
                type(dictionary['bytes']) is not int or
                not 0 < dictionary['bytes'] <= paddle_closure.MAX_DICTIONARY_BYTES or
                type(dictionary['sha256']) is not str or
                not re.fullmatch('[0-9a-f]{64}', dictionary['sha256'])):
            raise ValueError('Paddle rec dictionary identity/sha invalid')
    entries = closure['entries']
    if (not isinstance(entries, list) or
            not 2 <= len(entries) <= paddle_closure.MAX_SELECTED_FILES):
        raise ValueError('Paddle selected source members outside bounds')
    names = []
    total = 0
    for row in entries:
        if not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Paddle selected source member keys invalid')
        name = row['path']
        allowed = {paddle_closure._WEIGHTS, *paddle_closure._GRAPHS,
                   paddle_closure._OPTIONAL_INFO, *paddle_closure._CONFIGS,
                   *paddle_closure._INFER_YAMLS}
        if dictionary and name == dictionary['path']:
            pass
        elif type(name) is not str or name not in allowed:
            raise ValueError('Paddle selected source member path unapproved')
        if (type(row['bytes']) is not int or
                not 0 < row['bytes'] <= paddle_closure.MAX_TOTAL_BYTES or
                type(row['sha256']) is not str or
                not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Paddle selected source member bytes/SHA invalid')
        if dictionary and name == dictionary['path'] and (
                dictionary['bytes'] != row['bytes'] or
                dictionary['sha256'] != row['sha256']):
            raise ValueError('Paddle selected dictionary differs from entries')
        total += row['bytes']
        names.append(name)
    if (names != sorted(set(names)) or
            set(names) & set(paddle_closure._GRAPHS) != {closure['graph_file']} or
            paddle_closure._WEIGHTS not in names or
            (dictionary is not None and dictionary['path'] not in names) or
            (dictionary is None and any('/' in item for item in names)) or
            type(closure['file_count']) is not int or
            closure['file_count'] != len(entries) or
            type(closure['total_bytes']) is not int or
            total != closure['total_bytes'] or total > paddle_closure.MAX_TOTAL_BYTES):
        raise ValueError('Paddle selected graph/weight/count/byte inventory invalid')
    canonical = {key: closure[key] for key in (
        'schema', 'schema_version', 'role', 'graph_file', 'entries', 'dictionary')}
    import hashlib
    sha = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    if type(closure['tree_sha256']) is not str or sha != closure['tree_sha256']:
        raise ValueError('Paddle selected source tree SHA mismatch')
    return closure


def validate_selected_source_closure(closure):
    """Pure structural review of an ONNX+external_data recipe pin (not an OS lock)."""
    keys = frozenset((
        'source_kind', 'format', 'status', 'closure_schema',
        'source_closure_version', 'source_model', 'source_root', 'model_rel',
        'file_count', 'total_bytes', 'tree_sha256', 'dependencies', 'entries',
        'references', 'release_grade_immutable'))
    if not isinstance(closure, dict) or frozenset(closure) != keys:
        raise ValueError('selected ONNX source closure record has missing/unknown keys')
    if (closure['source_kind'] != 'selected_onnx_files' or
            closure['format'] != 'onnx' or
            closure['status'] != 'SELECTED_FILES_SCANNED_NOT_LOCKED' or
            closure['closure_schema'] != onnx_closure.SCHEMA or
            type(closure['source_closure_version']) is not int or
            closure['source_closure_version'] != onnx_closure.SCHEMA_VERSION or
            closure['release_grade_immutable'] is not False):
        raise ValueError('selected ONNX closure schema/version/status is invalid')
    source_model = closure['source_model']
    source_root = closure['source_root']
    if (not isinstance(source_model, str) or not os.path.isabs(source_model) or
            not isinstance(source_root, str) or
            os.path.dirname(source_model) != source_root or
            os.path.basename(source_model) != closure['model_rel'] or
            not source_model.lower().endswith('.onnx')):
        raise ValueError('selected ONNX closure original source identity is invalid')
    model_rel = onnx_closure._relative(closure['model_rel'], 'closure model filename')
    if '/' in model_rel:
        raise ValueError('selected model must be at private stage root')
    digest = closure['tree_sha256']
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('selected source closure SHA-256 is invalid')

    entries = closure['entries']
    if not isinstance(entries, list) or not entries or len(entries) > (
            onnx_closure.MAX_FILES * (onnx_closure.MAX_RELATIVE_DEPTH + 1)):
        raise ValueError('selected source closure entries exceed bounds')
    seen, folded, members = set(), set(), {}
    total = 0
    for item in entries:
        if (not isinstance(item, dict) or
                set(item) != {'path', 'kind', 'bytes', 'sha256'}):
            raise ValueError('selected source closure member is malformed')
        rel = onnx_closure._relative(item['path'], 'closure member')
        lowered = rel.casefold()
        if rel in seen or lowered in folded:
            raise ValueError('selected source closure duplicate/case-alias path')
        seen.add(rel)
        folded.add(lowered)
        if item['kind'] == 'directory':
            if type(item['bytes']) is not int or item['bytes'] != 0 or item['sha256'] is not None:
                raise ValueError('selected directory has nonempty bytes/hash')
        elif item['kind'] == 'file':
            if (type(item['bytes']) is not int or
                    not 0 < item['bytes'] <= onnx_closure.MAX_TOTAL_BYTES or
                    not isinstance(item['sha256'], str) or
                    not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
                raise ValueError('selected file has invalid size or SHA-256')
            total += item['bytes']
            if total > onnx_closure.MAX_TOTAL_BYTES:
                raise ValueError('selected source closure exceeds byte bound')
        else:
            raise ValueError('selected source closure member has invalid kind')
        members[rel] = item
    if [item['path'] for item in entries] != sorted(members):
        raise ValueError('selected source closure member order is not deterministic')
    file_names = {name for name in members
                  if members[name]['kind'] == 'file'}
    expected_dirs = {'/'.join(name.split('/')[:index])
                     for name in file_names
                     for index in range(1, len(name.split('/')))}
    if (set(members) - file_names != expected_dirs or
            any(members[name]['kind'] != 'directory' for name in expected_dirs) or
            model_rel not in file_names or len(file_names) < 2 or
            len(file_names) > onnx_closure.MAX_FILES):
        raise ValueError('selected closure contains missing/extra directories or model')
    dependencies = closure['dependencies']
    if (not isinstance(dependencies, list) or
            dependencies != sorted(file_names - {model_rel})):
        raise ValueError('selected ONNX sidecar list differs from selected members')
    if (type(closure['file_count']) is not int or
            closure['file_count'] != len(file_names) or
            type(closure['total_bytes']) is not int or
            closure['total_bytes'] != total or
            source_freeze._tree_sha({'entries': entries}) != digest):
        raise ValueError('selected closure count/byte extent/tree SHA differs')

    refs = closure['references']
    if (not isinstance(refs, list) or not refs or
            len(refs) > onnx_closure.MAX_EXTERNAL_REFERENCES):
        raise ValueError('selected ONNX has no bounded external tensor references')
    for ref in refs:
        if (not isinstance(ref, dict) or set(ref) != {
                'tensor', 'location', 'offset', 'length', 'expected_bytes',
                'dtype', 'dims'}):
            raise ValueError('selected external tensor record is malformed')
        if (not isinstance(ref['tensor'], str) or
                not isinstance(ref['location'], str) or
                ref['location'] not in dependencies or
                type(ref['offset']) is not int or ref['offset'] < 0 or
                (ref['length'] is not None and
                 (type(ref['length']) is not int or
                  not 0 < ref['length'] <= onnx_closure.MAX_TOTAL_BYTES)) or
                type(ref['expected_bytes']) is not int or
                not 0 < ref['expected_bytes'] <= onnx_closure.MAX_TOTAL_BYTES or
                type(ref['dtype']) is not int or
                ref['dtype'] not in onnx_closure._DTYPE_BYTES or
                not isinstance(ref['dims'], list) or
                any(type(dim) is not int or dim <= 0 for dim in ref['dims'])):
            raise ValueError('selected external tensor path/range/dtype is invalid')
        expected_bytes = onnx_closure._DTYPE_BYTES[ref['dtype']]
        for dim in ref['dims']:
            expected_bytes *= dim
            if expected_bytes > onnx_closure.MAX_TOTAL_BYTES:
                raise ValueError('external tensor dimensions exceed bounds')
        sidecar_size = members[ref['location']]['bytes']
        amount = (ref['length'] if ref['length'] is not None
                  else sidecar_size - ref['offset'])
        if (amount != ref['expected_bytes'] or
                amount != expected_bytes or
                amount > sidecar_size - ref['offset']):
            raise ValueError('external tensor range differs from selected sidecar')
    return closure


def validate_cpu_codegen_receipt(receipt, plan):
    """Validate a compiler-attribute observation, never a destination oracle."""
    if (not isinstance(plan, dict) or plan.get('route') != 'raw'
            or not isinstance(plan.get('cpu_codegen'), dict)):
        raise ValueError('CPU codegen receipt requires a selected RAW BuildPlan')
    keys = {'compiler_target', 'actual_tvm_target', 'binding',
            'destination_compatibility', 'llvm_triple_attested',
            'full_target_equivalence_attested'}
    if type(receipt) is not dict or set(receipt) != keys:
        raise ValueError('CPU codegen receipt fields missing or unexpected')
    if (type(receipt['compiler_target']) is not str or
            receipt['compiler_target'] != plan.get('compiler_target') or
            receipt['compiler_target'] != plan['cpu_codegen'].get('tvm_target') or
            receipt['binding'] != 'TVM_KIND_AND_MCPU_CHECKED_AFTER_EXPORT' or
            receipt['destination_compatibility'] != 'NOT_CHECKED' or
            receipt['llvm_triple_attested'] is not False or
            receipt['full_target_equivalence_attested'] is not False):
        raise ValueError('CPU receipt cannot claim unproved codegen or verification')
    actual = receipt['actual_tvm_target']
    if (type(actual) is not str or not 3 <= len(actual) <= 4096 or
            not actual.startswith('llvm') or
            any(ord(char) < 32 or ord(char) == 127 for char in actual)):
        raise ValueError('CPU receipt has invalid actual LLVM target')
    selected_mcpu = plan['cpu_codegen'].get('mcpu')
    found = re.findall(r'(?:^|\s)-mcpu=([^\s]+)', actual)
    if found != ([selected_mcpu] if selected_mcpu is not None else []):
        raise ValueError('CPU receipt actual mcpu differs from selected BuildPlan')
    return receipt


def build_recipe(source, target, shape, name, tune, trials_per_task, sm,
                 contract_input, provenance=None, license=None,
                 target_profile=None, build_plan=None,
                 effective_cpu_codegen=None):
    """Assemble the recipe dict from the inputs of one build.

    source           the model as the user named it - a URL or a path on disk.
                     The recipe records what was ASKED for, not the local file a
                     URL was downloaded to, so a repeat fetches the model afresh.
    target           cpu or cuda
    shape            the input-shape hint, e.g. "images:1,3,640,640", or None
    name             the module's base name
    tune             whether tuning was on (a bool; the tuning work directory is
                     an output location, not a build input, so it is not recorded
                     - a repeat places its own database)
    trials_per_task  trials each tuning task got, or None
    sm               the CUDA sm the target card is, e.g. "sm_89", or None
    contract_input   the model-contract dict build_contract_input produced, or
                     None when no task was given
    provenance       the audit facts _gather_provenance built (source_hash,
                     toolchain, built_at, ...), or None. The recipe is the input
                     for a CI repeat, and recording the source's own hash, the
                     toolchain that produced the module and the moment lets that
                     repeat verify it fetched the SAME model and know what made the
                     first one. These are provenance, not build knobs, so a repeat
                     does not feed them back in - it takes fresh ones of its own.
    license          the source model's licence, or None

    Returns a plain dict of JSON-native types, ready for write_recipe. The audit
    fields sit at the top level beside source (they describe the build as a whole,
    like source does), never inside the contract section, so contract_from_recipe
    is untouched and the recipe/manifest separation holds: none of these are the
    manifest's built-contract keys (task_kind, artifact, io, format_version).
    """
    sm = validate_target_sm(target, sm)
    if target_profile is not None:
        validate_recorded_profile(target_profile, target=target, req_sm=sm)
    if build_plan is not None:
        if target_profile is None:
            raise ValueError('BuildPlan requires a recorded target_profile')
        validate_build_plan(
            build_plan, source=source, target=target, req_sm=sm,
            target_profile=target_profile, shape=shape,
            tune=bool(tune), trials_per_task=trials_per_task,
            contract=contract_input)
    license = _checked_declared_model_license(license)
    facts = provenance or {}
    recipe = {
        "recipe_version": (
            RECIPE_PROFILE_VERSION if target_profile is not None
            else RECIPE_VERSION),
        "source": source,
        "source_hash": facts.get("source_hash"),
        "target": target,
        "sm": sm,
        "shape": shape,
        "name": name,
        "tune": bool(tune),
        "trials_per_task": trials_per_task,
        "toolchain": facts.get("toolchain"),
        "built_at": facts.get("built_at"),
        "license": license,
        "contract": contract_input,
    }
    
    
    
    if sm:
        recipe["tvm_target"] = tvm_target_spec(target, sm)
    if target_profile is not None:
        recipe['target_profile'] = target_profile
    if build_plan is not None:
        
        
        recipe['build_plan'] = build_plan
    if effective_cpu_codegen is not None:
        recipe['cpu_codegen_receipt_version'] = 1
        recipe['effective_cpu_codegen'] = validate_cpu_codegen_receipt(
            effective_cpu_codegen, build_plan)
    if facts.get('source_closure') is not None:
        
        
        
        recipe['source_closure'] = validate_selected_source_closure(
            facts['source_closure'])
    if facts.get('paddle_selected_source') is not None:
        if facts.get('source_closure') is not None:
            raise ValueError('Paddle and ONNX source closure cannot be mixed')
        if target_profile is None or build_plan is None:
            raise ValueError('Paddle source receipt requires a versioned BuildPlan')
        if build_plan.get('route') != 'raw':
            raise ValueError('Paddle selected source requires RAW converter route')
        recipe['paddle_selected_source'] = validate_paddle_selected_source(
            facts['paddle_selected_source'], source=source)
    return recipe


def write_recipe(recipe, path):
    """Write *recipe* to *path* as recipe.json - indented, UTF-8, LF endings.

    newline='' keeps the newline characters json emits from being doubled into
    CRLF on Windows, so the file is LF on every platform - the converter's own
    convention, shared with its report and manifest writers. A trailing newline
    terminates the last line.
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n")


def _unique_recipe_pairs(pairs):
    """Reject ambiguous JSON objects at any nesting depth on CLI replay."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate recipe JSON member: %s' % key)
        result[key] = value
    return result


def _reject_nonstandard_recipe_constant(value):
    raise ValueError('nonstandard JSON numeric constant in recipe: %s' % value)


def read_recipe(path):
    """Read a recipe.json back into a dict, with a soft version check.

    Soft means: the file must declare a recipe_version - that is what tells a
    recipe from any other JSON - but a version the reader does not recognise is
    accepted rather than refused. The fields this reader needs may well still be
    present, and turning a newer recipe away outright would help no one. A file
    that is not a JSON object, or that carries no recipe_version, is not a recipe
    and is refused by name rather than left to fail deep in a caller.
    """
    with open(path, encoding="utf-8") as handle:
        recipe = json.load(handle, object_pairs_hook=_unique_recipe_pairs,
                           parse_constant=_reject_nonstandard_recipe_constant)
    if not isinstance(recipe, dict):
        raise ValueError("a recipe must be a JSON object, not %s"
                         % type(recipe).__name__)
    if recipe.get("recipe_version") is None:
        raise ValueError("this file carries no recipe_version, so it is not a "
                         "build recipe: %s" % path)
    validate_family_license_receipt(recipe)
    if recipe.get('source_closure') is not None:
        if recipe.get('recipe_version', 0) < RECIPE_PROFILE_VERSION:
            raise ValueError('selected external ONNX closure requires recipe v2')
        closure = validate_selected_source_closure(recipe['source_closure'])
        raw_source = recipe.get('source')
        if (not isinstance(raw_source, str) or not raw_source or
                os.path.abspath(raw_source) != closure['source_model']):
            raise ValueError('saved external ONNX closure original source differs '
                             'from recipe input; refuse unsafe replay')
    if 'paddle_selected_source' in recipe:
        if (recipe.get('recipe_version', 0) < RECIPE_PROFILE_VERSION or
                'source_closure' in recipe):
            raise ValueError('Paddle selected source requires recipe v2 alone')
        selected = validate_paddle_selected_source(
            recipe['paddle_selected_source'], source=recipe.get('source'))
        plan = recipe.get('build_plan')
        if not isinstance(plan, dict) or plan.get('route') != 'raw':
            raise ValueError('Paddle selected recipe requires recorded RAW BuildPlan')
    if recipe.get("target_profile") is not None:
        if recipe['recipe_version'] < RECIPE_PROFILE_VERSION:
            raise ValueError('target_profile requires recipe_version %d'
                             % RECIPE_PROFILE_VERSION)
        validate_recorded_profile(
            recipe['target_profile'], target=recipe.get('target'),
            req_sm=recipe.get('sm'))
        requested = tvm_target_spec(
            recipe.get('target'), recipe.get('sm'))
        if recipe.get('tvm_target', requested) != requested:
            raise ValueError('recipe tvm_target differs from target_profile')
    if recipe.get('build_plan') is not None:
        if recipe.get('recipe_version', 0) < RECIPE_PROFILE_VERSION:
            raise ValueError('BuildPlan requires recipe_version %d'
                             % RECIPE_PROFILE_VERSION)
        validate_build_plan(
            recipe['build_plan'], source=recipe.get('source'),
            target=recipe.get('target'), req_sm=recipe.get('sm'),
            target_profile=recipe.get('target_profile'),
            shape=recipe.get('shape'), tune=recipe.get('tune'),
            trials_per_task=recipe.get('trials_per_task'),
            contract=recipe.get('contract'))
    if ('effective_cpu_codegen' in recipe or
            'cpu_codegen_receipt_version' in recipe):
        if (type(recipe.get('cpu_codegen_receipt_version')) is not int or
                recipe['cpu_codegen_receipt_version'] != 1 or
                'effective_cpu_codegen' not in recipe):
            raise ValueError('CPU compiler receipt version/content mismatch')
        validate_cpu_codegen_receipt(recipe['effective_cpu_codegen'],
                                     recipe.get('build_plan'))
    family = recipe.get('family_builder')
    if family is not None:
        if (not isinstance(family, dict) or
                set(family) != {'adapter', 'model_key', 'dtype'} or
                family.get('adapter') != 'llm' or
                not isinstance(family.get('model_key'), str) or
                not re.fullmatch(r'[a-z0-9][a-z0-9._-]*', family['model_key'])):
            raise ValueError('invalid saved native family builder contract')
        plan = recipe.get('build_plan')
        if (not isinstance(plan, dict) or
                plan.get('route') != 'family_full_build' or
                family.get('dtype') != plan.get('options', {}).get('dtype')):
            raise ValueError('saved family builder differs from BuildPlan')
    elif (isinstance(recipe.get('build_plan'), dict) and
          recipe['build_plan'].get('route') == 'family_full_build'):
        raise ValueError('native family BuildPlan has no model-family replay key')
    return recipe


def write_family_receipt(plan, output_directory, *, model_key, provenance,
                         license=None):
    """Persist a native LLM build request without forging a numeric passport.

    Shared by wizard and CLI. All four files use exclusive creation; there is
    still NO atomic whole-directory commit (that is a separate I-03 debt).
    """
    validate_build_plan(plan)
    if plan['route'] != 'family_full_build':
        raise ValueError('family receipt needs a family_full_build BuildPlan')
    if (not isinstance(model_key, str) or
            not re.fullmatch(r'[a-z0-9][a-z0-9._-]*', model_key)):
        raise ValueError('invalid family builder model_key')
    if not isinstance(provenance, dict):
        raise ValueError('native family import provenance must be an object')
    target = plan['target_profile']['deployment']
    options = plan['options']
    receipt = build_recipe(
        plan['source'], target['device'], None, model_key, False, None,
        target['sm'], None, target_profile=plan['target_profile'],
        build_plan=plan, license=license)
    receipt['family_builder'] = {
        'adapter': 'llm', 'model_key': model_key,
        'dtype': options['dtype'],
    }
    if receipt['license'] is not None:
        receipt['recipe_version'] = RECIPE_FAMILY_LICENSE_CLAIM_VERSION
        
        
        receipt['model_license_claim'] = _checked_model_license_claim({
            'schema': 'crosspascal.model-license-claim', 'version': 1,
            'license': receipt['license'],
            'basis': 'CALLER_DECLARED_NOT_ATTESTED',
            'redistribution_clearance': 'NOT_DETERMINED',
        }, receipt['license'])
    folder = os.path.abspath(output_directory)
    if not os.path.isdir(folder):
        raise ValueError('family output directory is absent: %s' % folder)
    
    
    
    
    from .llm_bundle_contract import validate_llm_output
    validate_llm_output(folder, plan, model_key, require_consumer=True)
    contents = {
        'recipe.json': json.dumps(receipt, indent=2, ensure_ascii=False) + '\n',
        'build-plan.json': json.dumps(plan, indent=2, ensure_ascii=False) + '\n',
        'import-provenance.json': json.dumps(provenance, indent=2,
                                            ensure_ascii=False) + '\n',
        'report.txt': (
            'Native family builder returned a bundle.\n'
            'Verification: NOT_CHECKED (no independent model oracle).\n'
            'Target: %s; requested compiler: %s\n'
            'RESULT: COMPILED / NOT VERIFIED\n'
            % (target['device'], plan['compiler_target'])),
    }
    existing = [name for name in contents if os.path.lexists(os.path.join(folder, name))]
    if existing:
        raise FileExistsError('family receipt would overwrite existing outputs: '
                              + ', '.join(sorted(existing)))
    for name, text in contents.items():
        with open(os.path.join(folder, name), 'x', encoding='utf-8',
                  newline='\n') as handle:
            handle.write(text)
    return receipt


def contract_from_recipe(recipe):
    """Rebuild the pipeline.run contract from a recipe's saved contract section.

    A recipe stores the very dict build_contract_input produced. Feeding its
    fields back through that same builder returns an equal dict AND re-applies the
    builder's own gate - a task needs an adapter - so a hand-edited recipe cannot
    smuggle a half-contract into a repeat. A recipe with no contract (no task was
    given at build time) rebuilds to None, which is the no-manifest build.
    """
    saved = recipe.get("contract")
    if not saved:
        return None
    return build_contract_input(
        saved.get("task") or "",
        saved.get("adapter") or "",
        adapter_api_version=saved.get("adapter_api_version", 1),
        layout=saved.get("layout") or "NCHW",
        score_threshold=saved.get("score_threshold"),
        nms_threshold=saved.get("nms_threshold"),
        mean=saved.get("mean"),
        std=saved.get("std"),
        keypoint_count=saved.get("keypoint_count"),
        labels=saved.get("labels"))
