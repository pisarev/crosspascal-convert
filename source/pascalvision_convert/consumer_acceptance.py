# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Aggregate declared LLM checks without turning missing evidence into success."""

from .llm_build_profile import QWEN_CUDA_PROFILE

SCHEMA = "crosspascal.consumer-checks.v1"
STATUSES = frozenset(("PASS", "FAIL", "NOT_RUN", "ERROR"))
REQUIRED = (
    "build_completed",
    "bundle_abi",
    "selected_source",
    "selected_target",
    "consumer_runtime",
)
PROFILES = frozenset(("native-llm-build-v1", QWEN_CUDA_PROFILE))


def aggregate(profile_id, checks):
    if profile_id not in PROFILES:
        raise ValueError("Unknown consumer acceptance profile")
    if type(checks) is not list:
        raise ValueError("Consumer checks must be a list")
    seen = {}
    observations = []
    for check in checks:
        if (type(check) is not dict or type(check.get("id")) is not str or
                check.get("status") not in STATUSES or
                check.get("role") not in ("required", "diagnostic")):
            raise ValueError("Malformed consumer check")
        name = check["id"]
        if name in seen:
            raise ValueError("Duplicate consumer check: " + name)
        if (name in REQUIRED) != (check["role"] == "required"):
            raise ValueError("Check role differs from the declared profile: " + name)
        seen[name] = check
        observations.append(dict(check))
    missing = [name for name in REQUIRED if name not in seen]
    for name in missing:
        check = {"id": name, "role": "required", "status": "NOT_RUN",
                 "detail": "No result was supplied for this required check."}
        seen[name] = check
        observations.append(check)
    statuses = [seen[name]["status"] for name in REQUIRED]
    status = ("ERROR" if "ERROR" in statuses else
              "FAIL" if "FAIL" in statuses else
              "NOT_RUN" if "NOT_RUN" in statuses else "PASS")
    return {
        "schema": SCHEMA,
        "profile_id": profile_id,
        "status": status,
        "required_checks_pass": status == "PASS",
        "missing_checks": missing,
        "checks": observations,
        "diagnostic_failures": [
            item["id"] for item in observations
            if item["role"] == "diagnostic" and item["status"] in ("FAIL", "ERROR")
        ],
        "scope": "The declared model, build options, inputs, and consumer run only.",
    }


def after_native_build(plan):
    profile = plan["options"].get("llm")
    profile_id = profile["profile_id"] if profile else "native-llm-build-v1"
    checks = [
        {"id": name, "role": "required", "status": "PASS",
         "detail": "Checked by the native build and its output/source validators."}
        for name in REQUIRED if name != "consumer_runtime"
    ]
    checks.append({
        "id": "consumer_runtime", "role": "required", "status": "NOT_RUN",
        "detail": "Built; run the included Delphi or Lazarus example to verify.",
    })
    return aggregate(profile_id, checks)

