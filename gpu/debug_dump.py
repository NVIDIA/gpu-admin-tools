#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import datetime
import json
import posixpath
import socket
import time
import zipfile
from dataclasses import dataclass
from typing import Any
from gpu.regs.core import ArrayMetadata, RegisterMetadata
from gpu.unit import DEBUG_DUMP_CAPABILITY

_FORMAT_VERSION = 1
_DEVICES_DIR = "devices"


@dataclass(frozen=True)
class SampleConfig:
    count: int = 1
    interval_ms: int = 0


@dataclass(frozen=True)
class CaptureOptions:
    pass


@dataclass(frozen=True)
class CapturedRecord:
    record: dict[str, Any]
    reg: Any = None

    def __getitem__(self, key):
        return self.record[key]

    def get(self, key, default=None):
        return self.record.get(key, default)

    def __contains__(self, key):
        return key in self.record


class DeviceDumpCapture:
    def __init__(self, gpu):
        self.gpu = gpu
        self.started_at = time.perf_counter()
        self.units: dict[str, dict[str, Any]] = {}
        self.attachments: list[dict[str, str]] = []

    def _unit_bucket(self, unit_name):
        return self.units.setdefault(unit_name, {"registers": {}, "data": {}})

    def add_register_record(self, unit_name, key, record):
        self._add_unit_record(unit_name, "registers", key, record)

    def add_data_record(self, unit_name, key, record):
        self._add_unit_record(unit_name, "data", key, record)

    def _add_unit_record(self, unit_name, section, key, record):
        bucket = self._unit_bucket(unit_name)[section]
        existing = bucket.get(key)
        if existing is None:
            bucket[key] = record
            return
        bucket[key] = _merge_record_variants(existing, record)

    def add_attachment(self, unit_name, path, description, content):
        self.attachments.append({
            "unit": unit_name,
            "path": path,
            "description": description,
            "content": content,
        })

    @property
    def interesting_records(self):
        records = []
        for unit_name, unit_data in self.units.items():
            for section in ["registers", "data"]:
                for key, record_or_records in unit_data.get(section, {}).items():
                    for record in _record_variants(record_or_records):
                        interesting = record.get("interesting")
                        if interesting is None:
                            continue
                        summary = {
                            "unit": unit_name,
                            "key": key,
                            "reason": interesting.get("reason", "interesting"),
                        }
                        if "instance" in record:
                            summary["instance"] = record["instance"]
                        records.append(summary)
        return records

    def elapsed_ms(self):
        return int((time.perf_counter() - self.started_at) * 1000)

    def to_json(self):
        device = self.gpu
        attachment_meta = []
        for attachment in self.attachments:
            attachment_meta.append({
                "unit": attachment["unit"],
                "path": attachment["path"],
                "description": attachment["description"],
            })

        interesting = self.interesting_records
        return {
            "format_version": _FORMAT_VERSION,
            "device": {
                "name": device.name,
                "arch": device.arch,
                "chip": device.chip,
                "bdf": device.bdf,
                "pci_device_id": _hex_int(device.device, width=4),
                "pci_vendor_id": _hex_int(device.vendor, width=4),
                "pci_subsystem_vendor_id": _hex_int(device.svid, width=4),
                "pci_subsystem_device_id": _hex_int(device.ssid, width=4),
                "bar0_addr": _hex_int(device.bar0_addr),
            },
            "capture": {
                "elapsed_ms": self.elapsed_ms(),
            },
            "summary": {
                "interesting_count": len(interesting),
            },
            "units": self.units,
            "interesting_records": interesting,
            "attachments": attachment_meta,
        }


class UnitDumpCapture:
    def __init__(self, device_capture, unit_name, options):
        self.device_capture = device_capture
        self.gpu = device_capture.gpu
        self.unit_name = unit_name
        self.options = options

    def subunit(self, name):
        return UnitDumpCapture(self.device_capture, f"{self.unit_name}.{name}", self.options)

    def register(
        self,
        register,
        key=None,
        base=0,
        samples=None,
        interval_ms=None,
        instance=None,
        interesting=None,
        sample_aggregation=None,
        debug_dump_metadata=None,
    ):
        if isinstance(register, ArrayMetadata):
            return self.array(
                register,
                key=key,
                base=base,
                samples=samples,
                interval_ms=interval_ms,
                instance=instance,
                interesting=interesting,
                debug_dump_metadata=debug_dump_metadata,
            )
        try:
            record = _capture_register(
                self.gpu,
                register,
                started_at=self.device_capture.started_at,
                base=base,
                samples=samples,
                interval_ms=interval_ms,
                instance=instance,
                sample_aggregation=sample_aggregation,
            )
        except Exception as err:  # pylint: disable=broad-except
            record = _capture_error_record(
                kind="register",
                started_at=self.device_capture.started_at,
                error=err,
                offset=register.address,
                base=base,
                instance=instance,
            )
        _set_interesting_from_rule(record, interesting)
        _set_record_debug_dump_metadata(record, debug_dump_metadata)
        self.device_capture.add_register_record(self.unit_name, register.name if key is None else key, record)
        return CapturedRecord(record=record, reg=_captured_register_value(register, record))

    def registers(self, registers, base=0, samples=None, interval_ms=None, instance=None, sample_aggregation=None):
        return [
            self.register(
                register,
                base=base,
                samples=samples,
                interval_ms=interval_ms,
                instance=instance,
                sample_aggregation=sample_aggregation,
            )
            for register in registers
        ]

    def array(
        self,
        array,
        key=None,
        base=0,
        samples=None,
        interval_ms=None,
        instance=None,
        interesting=None,
        debug_dump_metadata=None,
    ):
        try:
            record = _capture_array(
                self.gpu,
                array,
                started_at=self.device_capture.started_at,
                base=base,
                samples=samples,
                interval_ms=interval_ms,
                instance=instance,
            )
        except Exception as err:  # pylint: disable=broad-except
            record = _capture_error_record(
                kind="array",
                started_at=self.device_capture.started_at,
                error=err,
                offset=array.address,
                base=base,
                instance=instance,
                stride=array.stride,
                size=array.size,
            )
        _set_interesting_from_rule(record, interesting)
        _set_record_debug_dump_metadata(record, debug_dump_metadata)
        self.device_capture.add_register_record(self.unit_name, array.name if key is None else key, record)
        return CapturedRecord(record=record)

    def module(self, register_module, base=0, instance=None):
        module_defaults = getattr(register_module, "GPU_REGS_DEBUG_DUMP_DEFAULTS", {})
        records = []
        for entity in _iter_module_entities(register_module):
            metadata = _merge_debug_dump_metadata(module_defaults, getattr(entity, "debug_dump", None))
            if metadata.get("exclude"):
                continue
            if isinstance(entity, ArrayMetadata):
                if metadata.get("include") is False:
                    continue
                if entity.size <= 0:
                    continue
                records.append(
                    self.array(
                        entity,
                        base=base,
                        samples=metadata.get("samples"),
                        interval_ms=metadata.get("interval_ms"),
                        instance=instance,
                        interesting=metadata.get("interesting"),
                        debug_dump_metadata=metadata,
                    )
                )
                continue
            if metadata.get("include") is False:
                continue
            records.append(
                self.register(
                    entity,
                    base=base,
                    samples=metadata.get("samples"),
                    interval_ms=metadata.get("interval_ms"),
                    instance=instance,
                    interesting=metadata.get("interesting"),
                    sample_aggregation=metadata.get("aggregation"),
                    debug_dump_metadata=metadata,
                )
            )
        return records

    def offset(self, key, offset, samples=None, interval_ms=None, instance=None, interesting=None):
        try:
            record = _capture_offset(
                self.gpu,
                offset,
                started_at=self.device_capture.started_at,
                samples=samples,
                interval_ms=interval_ms,
                instance=instance,
            )
        except Exception as err:  # pylint: disable=broad-except
            record = _capture_error_record(
                kind="offset",
                started_at=self.device_capture.started_at,
                error=err,
                offset=offset,
                instance=instance,
            )
        _set_interesting_from_rule(record, interesting)
        self.device_capture.add_data_record(self.unit_name, key, record)
        return CapturedRecord(record=record)

    def data(self, key, value, samples=None, interval_ms=None, interesting=None):
        record = _capture_data(
            value,
            started_at=self.device_capture.started_at,
            samples=samples,
            interval_ms=interval_ms,
        )
        _set_interesting_from_rule(record, interesting)
        self.device_capture.add_data_record(self.unit_name, key, record)
        return CapturedRecord(record=record)

    def data_pairs(self, key, pairs, samples=None, interval_ms=None, interesting=None):
        record = _capture_data_pairs(
            pairs,
            started_at=self.device_capture.started_at,
            samples=samples,
            interval_ms=interval_ms,
        )
        _set_interesting_from_rule(record, interesting)
        self.device_capture.add_data_record(self.unit_name, key, record)
        return CapturedRecord(record=record)

    def attachment(self, path, description, content):
        self.device_capture.add_attachment(self.unit_name, path, description, content)

    def set_interesting(self, record, interesting):
        _set_record_interesting(_captured_record_dict(record), interesting)
        return record


def _record_instance(record):
    return record.get("instance")


def _iter_module_entities(register_module):
    entities = []
    for entity in vars(register_module).values():
        if isinstance(entity, RegisterMetadata):
            entities.append(entity)
    return sorted(entities, key=lambda entity: (entity.address, entity.name))


def _merge_debug_dump_metadata(*metadatas):
    merged = {}
    for metadata in metadatas:
        if not metadata:
            continue
        for key, value in metadata.items():
            if key == "dimensions" and isinstance(value, dict):
                merged.setdefault("dimensions", {})
                merged["dimensions"].update(value)
            elif key == "tags":
                merged["tags"] = _merge_debug_dump_tags(merged.get("tags"), value)
            else:
                merged[key] = value
    return merged


def _merge_debug_dump_tags(*tag_lists):
    tags = []
    seen = set()
    for tag_list in tag_lists:
        for tag in _debug_dump_tags(tag_list):
            if tag in seen:
                continue
            tags.append(tag)
            seen.add(tag)
    return tags


def _debug_dump_tags(tag_list):
    if tag_list is None:
        return []
    if isinstance(tag_list, str):
        return [tag_list]
    return list(tag_list)


def _record_variants(record_or_records):
    if isinstance(record_or_records, list):
        return record_or_records
    return [record_or_records]


def _merge_record_variants(existing, record):
    record_instance = _record_instance(record)
    if isinstance(existing, list):
        for index, existing_record in enumerate(existing):
            if _record_instance(existing_record) == record_instance:
                existing[index] = record
                return existing
        existing.append(record)
        return existing

    if _record_instance(existing) == record_instance:
        return record

    return [existing, record]


def _capture_gpu_units(gpu, device_capture, options):
    gpu.ensure_units_with_capability(DEBUG_DUMP_CAPABILITY)

    for unit_name, unit in list(gpu.units.items()):
        try:
            unit.debug_dump_capture(UnitDumpCapture(device_capture, unit_name, options), options)
        except Exception as err:  # pylint: disable=broad-except
            error_text = f"{type(err).__name__}: {err}"
            record = _capture_data({
                "status": "error",
                "error": error_text,
            }, started_at=device_capture.started_at)
            _set_record_interesting(record, error_text)
            device_capture.add_data_record(unit_name, "__capture_error__", record)


def default_output_prefix(now=None, hostname=None):
    if now is None:
        now = datetime.datetime.now()
    if hostname is None:
        hostname = socket.gethostname().split(".")[0]
    return f"{hostname}-{now.strftime('%Y-%m-%d-%H-%M-%S')}-debug-dump"


def capture_gpu_bundle(gpus, output_file, options: CaptureOptions):
    if not gpus:
        raise ValueError("debug-dump capture requires at least one GPU")

    manifest = {
        "format_version": _FORMAT_VERSION,
        "bundle_kind": "raw",
        "tool": "nvidia_gpu_tools",
        "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "files": [],
        "summary": {
            "device_count": len(gpus),
            "interesting_count": 0,
        },
    }

    out_path = output_file if output_file.endswith(".zip") else f"{output_file}.zip"

    device_payloads = []
    attachments = []

    for gpu in gpus:
        capture = DeviceDumpCapture(gpu)
        _capture_gpu_units(gpu, capture, options)
        device_path = posixpath.join(_DEVICES_DIR, f"{_safe_bdf(gpu)}.json")
        payload = capture.to_json()
        device_payloads.append((device_path, payload))
        for attachment in capture.attachments:
            attachments.append(attachment)
        manifest["summary"]["interesting_count"] += payload["summary"]["interesting_count"]
        manifest["files"].append({
            "type": "device",
            "path": device_path,
            "bdf": gpu.bdf,
        })

    for attachment in attachments:
        manifest["files"].append({
            "type": "attachment",
            "path": attachment["path"],
            "unit": attachment["unit"],
            "description": attachment["description"],
        })

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", _json_dumps(manifest))
        for path, payload in device_payloads:
            bundle.writestr(path, _json_dumps(payload))
        for attachment in attachments:
            bundle.writestr(attachment["path"], attachment["content"])

    return out_path


def _sample_config(samples=None, interval_ms=None):
    return SampleConfig(
        count=1 if samples is None else samples,
        interval_ms=0 if interval_ms is None else interval_ms,
    )


def _set_interesting_from_rule(record, interesting_rule):
    if record is None or interesting_rule is None or "interesting" in record:
        return
    interesting = _evaluate_interesting_rule(record, interesting_rule)
    _set_record_interesting(record, interesting)


def _evaluate_interesting_rule(record, interesting_rule):
    if callable(interesting_rule):
        return interesting_rule(_record_sample_values(record))
    if not isinstance(interesting_rule, dict) or "rule" not in interesting_rule:
        return interesting_rule

    rule = interesting_rule["rule"]
    if rule == "nonzero":
        if not _record_has_nonzero_value(record):
            return None
    elif rule == "nonzero_or_unreadable":
        if not _record_has_nonzero_value(record) and not _record_has_unreadable_value(record):
            return None
    else:
        raise ValueError(f"unknown interesting rule: {rule}")

    interesting = dict(interesting_rule)
    interesting.pop("rule", None)
    return interesting


def _captured_record_dict(record):
    if isinstance(record, CapturedRecord):
        return record.record
    return record


def _captured_register_value(register, record):
    if record.get("kind") != "register":
        return None
    if record.get("status") != "ok":
        return None
    raw_value = record.get("value")
    if not isinstance(raw_value, str) or not raw_value.startswith("0x"):
        return None
    return register.value_from_int(int(raw_value, 16))


def _record_sample_values(record):
    values = []
    for value, _status, count in _record_sample_value_entries(record):
        values.extend([value] * count)
    return values


def _record_has_nonzero_value(record):
    for value, status, _count in _record_sample_value_entries(record):
        if status == "unreadable":
            continue
        if _is_nonzero_sample_value(value):
            return True
    return False


def _record_has_unreadable_value(record):
    return any(status == "unreadable" for _value, status, _count in _record_sample_value_entries(record))


def _record_sample_value_entries(record):
    entries = []
    for sample in _record_samples(record):
        count = sample.get("count", 1)
        if "value" in sample:
            entries.append((
                _normalize_sample_value(sample["value"]),
                sample.get("status"),
                count,
            ))
        for entry in sample.get("entries", []):
            if "value" not in entry:
                continue
            entries.append((
                _normalize_sample_value(entry["value"]),
                entry.get("status"),
                count,
            ))
    return entries


def _normalize_sample_value(value):
    if isinstance(value, str) and value.startswith("0x"):
        try:
            return int(value, 16)
        except ValueError:
            return value
    return value


def _is_nonzero_sample_value(value):
    if isinstance(value, int):
        return value != 0
    return False


def _capture_register(gpu, register, started_at, base=0, samples=None, interval_ms=None, instance=None, sample_aggregation=None):
    sample_config = _sample_config(samples=samples, interval_ms=interval_ms)
    record = {
        "kind": "register",
        "offset": _hex_int(register.address),
    }
    if base != 0:
        record["base"] = _hex_int(base)
    if instance is not None:
        record["instance"] = instance

    collected = []
    for index in range(sample_config.count):
        raw_value = gpu.regs.read(register, base=base).value
        sample = {
            "index": index,
            "elapsed_ms": _elapsed_ms_since(started_at),
        }
        sample["status"] = "unreadable" if _is_badf_read(raw_value) else "ok"
        sample["value"] = _hex_int(raw_value)
        collected.append(sample)
        _sleep_between_samples(index, sample_config)

    return _finalize_sampled_record(record, sample_config, collected, sample_aggregation=sample_aggregation)


def _capture_array(gpu, array, started_at, base=0, samples=None, interval_ms=None, instance=None):
    sample_config = _sample_config(samples=samples, interval_ms=interval_ms)
    record = {
        "kind": "array",
        "offset": _hex_int(array.address),
        "stride": _hex_int(array.stride),
        "size": array.size,
    }
    if base != 0:
        record["base"] = _hex_int(base)
    if instance is not None:
        record["instance"] = instance

    collected = []
    for sample_index in range(sample_config.count):
        sample = {
            "index": sample_index,
            "elapsed_ms": _elapsed_ms_since(started_at),
            "entries": [],
        }
        for entry_index in range(array.size):
            raw_value = gpu.regs.read(array(entry_index), base=base).value
            entry = {
                "status": "unreadable" if _is_badf_read(raw_value) else "ok",
                "value": _hex_int(raw_value),
            }
            sample["entries"].append(entry)
        collected.append(sample)
        _sleep_between_samples(sample_index, sample_config)

    return _finalize_sampled_record(record, sample_config, collected)


def _capture_offset(gpu, offset, started_at, samples=None, interval_ms=None, instance=None):
    sample_config = _sample_config(samples=samples, interval_ms=interval_ms)
    record = {
        "kind": "offset",
        "offset": _hex_int(offset),
    }
    if instance is not None:
        record["instance"] = instance

    collected = []
    for index in range(sample_config.count):
        raw_value = gpu.read_bad_ok(offset)
        sample = {
            "index": index,
            "elapsed_ms": _elapsed_ms_since(started_at),
            "value": _hex_int(raw_value),
            "status": "unreadable" if _is_badf_read(raw_value) else "ok",
        }
        collected.append(sample)
        _sleep_between_samples(index, sample_config)

    return _finalize_sampled_record(record, sample_config, collected)


def _capture_data(value_or_collector, started_at, samples=None, interval_ms=None):
    sample_config = _sample_config(samples=samples, interval_ms=interval_ms)
    record = {
        "kind": "custom",
    }

    collected = []
    for index in range(sample_config.count):
        try:
            value = value_or_collector() if callable(value_or_collector) else value_or_collector
        except Exception as err:  # pylint: disable=broad-except
            value = {
                "status": "error",
                "error": str(err),
            }
        sample = {
            "index": index,
            "elapsed_ms": _elapsed_ms_since(started_at),
            "value": _normalize_data_value(value),
        }
        collected.append(sample)
        _sleep_between_samples(index, sample_config)
    return _finalize_sampled_record(record, sample_config, collected)


def _capture_data_pairs(pairs_or_collector, started_at, samples=None, interval_ms=None):
    sample_config = _sample_config(samples=samples, interval_ms=interval_ms)
    record = {
        "kind": "custom_list",
    }

    collected = []
    for index in range(sample_config.count):
        sample = {
            "index": index,
            "elapsed_ms": _elapsed_ms_since(started_at),
        }
        try:
            pairs = pairs_or_collector() if callable(pairs_or_collector) else pairs_or_collector
            sample["entries"] = _normalize_data_pair_entries(pairs)
        except Exception as err:  # pylint: disable=broad-except
            sample["status"] = "error"
            sample["error"] = str(err)
            sample["value"] = {
                "status": "error",
                "error": str(err),
            }
        collected.append(sample)
        _sleep_between_samples(index, sample_config)
    return _finalize_sampled_record(record, sample_config, collected)


def _elapsed_ms_since(started_at):
    return int((time.perf_counter() - started_at) * 1000)


def _normalize_data_value(value):
    if isinstance(value, dict):
        return {str(k): _normalize_data_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_normalize_data_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_data_value(v) for v in value]
    return value


def _normalize_data_pair_entries(pairs):
    entries = []
    for pair in pairs:
        if isinstance(pair, dict):
            entry = {
                "name": str(pair["name"]),
                "value": _normalize_data_value(pair["value"]),
            }
            if "key" in pair:
                entry["key"] = str(pair["key"])
            entries.append(entry)
            continue

        name, value = pair
        entries.append({
            "name": str(name),
            "value": _normalize_data_value(value),
        })
    return entries


def _sleep_between_samples(index, sample_config):
    if index + 1 >= sample_config.count:
        return
    if sample_config.interval_ms <= 0:
        return
    time.sleep(sample_config.interval_ms / 1000.0)


def _set_record_interesting(record, interesting):
    if interesting is None or "interesting" in record:
        return
    if isinstance(interesting, str):
        interesting = {"reason": interesting}
    record["interesting"] = interesting


def _set_record_debug_dump_metadata(record, metadata):
    if not metadata:
        return
    record_metadata = {}
    for key in ("dimensions", "order", "display_name", "path", "tags"):
        if key not in metadata:
            continue
        value = metadata[key]
        if key == "dimensions" and isinstance(value, dict):
            value = dict(value)
        elif key == "tags":
            value = _debug_dump_tags(value)
            if not value:
                continue
        record_metadata[key] = value
    if record_metadata:
        record["debug_dump"] = record_metadata


def _record_samples(record):
    if "samples" in record:
        return record["samples"]
    sample = {}
    for key in ["index", "elapsed_ms", "status", "value", "error", "decoded", "entries"]:
        if key in record:
            sample[key] = record[key]
    if sample:
        return [sample]
    return []


def _finalize_sampled_record(record, sample_config, samples, sample_aggregation=None):
    if sample_config.count == 1 and sample_config.interval_ms == 0 and len(samples) == 1:
        record.update(_inline_sample_fields(samples[0]))
        return record

    record["sample_config"] = {
        "count": sample_config.count,
        "interval_ms": sample_config.interval_ms,
    }
    if sample_aggregation is not None:
        record["sample_config"]["aggregation"] = sample_aggregation
    if sample_aggregation == "histogram":
        record["samples"] = _histogram_sample_bins(samples)
        return record
    record["samples"] = samples
    return record


def _inline_sample_fields(sample):
    return {
        key: value
        for key, value in sample.items()
        if key != "index"
    }


def _capture_error_record(kind, started_at, error, offset=None, base=0, instance=None, stride=None, size=None):
    error_text = f"{type(error).__name__}: {error}"
    record = {
        "kind": kind,
        "elapsed_ms": _elapsed_ms_since(started_at),
        "status": "error",
        "error": error_text,
    }
    if offset is not None:
        record["offset"] = _hex_int(offset)
    if base != 0:
        record["base"] = _hex_int(base)
    if instance is not None:
        record["instance"] = instance
    if stride is not None:
        record["stride"] = _hex_int(stride)
    if size is not None:
        record["size"] = size
    _set_record_interesting(record, error_text)
    return record


def _histogram_sample_bins(samples):
    histogram = {}
    for sample in samples:
        bucket_key = (sample["status"], sample["value"])
        bucket = histogram.get(bucket_key)
        if bucket is None:
            histogram[bucket_key] = {
                "status": sample["status"],
                "value": sample["value"],
                "count": 1,
            }
            continue
        bucket["count"] += 1
    return sorted(
        histogram.values(),
        key=lambda bucket: (-bucket["count"], bucket["status"], bucket["value"]),
    )


def _hex_int(value, width=0):
    if width > 0:
        return f"0x{value:0{width}x}"
    return f"0x{value:x}"


def _is_badf_read(raw_value):
    # BAR0 bus error sentinel: reads to unmapped/fenced registers return 0xBADFxxxx
    return (raw_value >> 16) == 0xBADF


def _safe_bdf(gpu):
    return gpu.bdf.replace(":", "_").replace(".", "_")


def _json_dumps(payload):
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True)
