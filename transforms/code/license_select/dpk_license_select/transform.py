# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################


import json
from argparse import ArgumentParser, Namespace

import pyarrow as pa
import pyarrow.compute as pc
from data_processing.data_access import DataAccess, DataAccessFactory
from data_processing.transform import AbstractTableTransform, TransformConfiguration
from data_processing.utils import (
    CLIArgumentProvider,
    TransformUtils,
    get_dpk_logger,
    str2bool,
)
from dpk_license_select.transformer import (
    AllowLicenseStatusTransformer,
    DenyLicenseStatusTransformer,
)


logger = get_dpk_logger()

LICENSE_SELECT_PARAMS = "license_select_params"

shortname = "lc"
CLI_PREFIX = f"{shortname}_"

LICENSE_COLUMN_NAME_KEY = "license_column_name"
LICENSE_COLUMN_NAME_CLI_KEY = f"{CLI_PREFIX}{LICENSE_COLUMN_NAME_KEY}"

DENY_LICENSES_KEY = "deny_licenses"
DENY_LICENSES_CLI_KEY = f"{CLI_PREFIX}{DENY_LICENSES_KEY}"

LICENSES_FILE_KEY = "licenses_file"
LICENSES_FILE_CLI_KEY = f"{CLI_PREFIX}{LICENSES_FILE_KEY}"

ALLOW_NO_LICENSE_KEY = "allow_no_license"
ALLOW_NO_LICENSE_CLI_KEY = f"{CLI_PREFIX}{ALLOW_NO_LICENSE_KEY}"

LICENSE_COLUMN_DEFAULT = "license"
LICENSES_KEY = "licenses"

# Statistics keys emitted per-table by transform(). These are all flat integers so
# DPK's stats aggregation (which sums values per key across files) merges them
# correctly. Percentages and the per-license breakdown are derived from these in
# the runtime's compute_execution_stats (a dict value here would break aggregation).
STATS_TOTAL_DOCS = "total_docs"
STATS_DOCS_APPROVED = "docs_approved"
STATS_DOCS_REJECTED = "docs_rejected"
STATS_DOCS_NO_LICENSE = "docs_no_license"
# Per-license flat keys: "<prefix><delim><license-id>". License ids never contain
# '::', so the delimiter is safe to split on later when reshaping into a nested map.
STATS_LIC_DELIM = "::"
STATS_LIC_KEPT_PREFIX = f"lic_kept{STATS_LIC_DELIM}"
STATS_LIC_REJECTED_PREFIX = f"lic_rejected{STATS_LIC_DELIM}"


def _license_stats(table: pa.Table) -> dict:
    """
    Compute flat per-table statistics from a transformed table that has a
    ``license`` column (string or list<string>) and a boolean ``license_status``
    column. Returns a dict of flat integer keys (see STATS_* constants).

    Counting rules:
      * total_docs / docs_approved / docs_rejected come from license_status.
      * docs_no_license counts rows with no license: a null (string column) or a
        null/empty list (list column).
      * Per-license keys count, for each license id a row lists, whether that row
        was kept (license_status true) or rejected. A multi-license row increments
        every license it lists; rows with no license contribute only to
        docs_no_license, never to a per-license key.
    """
    status = table.column("license_status")
    licenses = table.column("license")

    total = table.num_rows
    approved = pc.sum(pc.cast(status, pa.int64())).as_py() or 0
    rejected = total - approved

    stats = {
        STATS_TOTAL_DOCS: total,
        STATS_DOCS_APPROVED: int(approved),
        STATS_DOCS_REJECTED: int(rejected),
    }

    is_list = pa.types.is_list(licenses.type) or pa.types.is_large_list(licenses.type)
    status_list = status.to_pylist()
    license_list = licenses.to_pylist()

    no_license = 0
    kept_counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    for lic_value, kept in zip(license_list, status_list):
        if is_list:
            ids = lic_value if lic_value else []
        else:
            ids = [lic_value] if lic_value is not None else []
        if len(ids) == 0:
            no_license += 1
            continue
        bucket = kept_counts if kept else rejected_counts
        for lic in ids:
            if lic is None:
                # a null element inside a list — treat as a no-license signal for
                # that entry; do not create a per-license key for None.
                continue
            # Group case-insensitively so "MIT"/"mit" don't fragment — matches the
            # case-insensitive approval logic in transformer.py.
            lic = str.casefold(lic)
            bucket[lic] = bucket.get(lic, 0) + 1

    stats[STATS_DOCS_NO_LICENSE] = no_license
    for lic, count in kept_counts.items():
        stats[f"{STATS_LIC_KEPT_PREFIX}{lic}"] = count
    for lic, count in rejected_counts.items():
        stats[f"{STATS_LIC_REJECTED_PREFIX}{lic}"] = count
    return stats


def compute_license_percentages(stats: dict) -> dict:
    """
    Post-aggregation reshape: given the fully-merged flat stats dict (summed across
    all files), add percentage fields and fold the flat per-license keys into a
    nested ``per_license_stats`` map. Mutates and returns ``stats``.

    Shared by the pure-Python and Ray runtimes so the two stay identical.

    Adds:
      * pct_docs_approved / pct_docs_rejected / pct_docs_no_license (% of total_docs)
      * per_license_stats: { "<license>": {kept, rejected, total, pct_of_corpus} }
        where pct_of_corpus = total / total_docs * 100
    The raw lic_kept::* / lic_rejected::* keys are removed after reshaping.
    """
    total = stats.get(STATS_TOTAL_DOCS, 0) or 0

    def pct(n: int) -> float:
        return round(100.0 * n / total, 2) if total > 0 else 0.0

    stats["pct_docs_approved"] = pct(stats.get(STATS_DOCS_APPROVED, 0))
    stats["pct_docs_rejected"] = pct(stats.get(STATS_DOCS_REJECTED, 0))
    stats["pct_docs_no_license"] = pct(stats.get(STATS_DOCS_NO_LICENSE, 0))

    per_license: dict[str, dict] = {}
    for key in [k for k in stats if k.startswith(STATS_LIC_KEPT_PREFIX) or k.startswith(STATS_LIC_REJECTED_PREFIX)]:
        if key.startswith(STATS_LIC_KEPT_PREFIX):
            lic = key[len(STATS_LIC_KEPT_PREFIX):]
            field = "kept"
        else:
            lic = key[len(STATS_LIC_REJECTED_PREFIX):]
            field = "rejected"
        entry = per_license.setdefault(lic, {"kept": 0, "rejected": 0})
        entry[field] += stats.pop(key)

    for lic, entry in per_license.items():
        entry["total"] = entry["kept"] + entry["rejected"]
        entry["pct_of_corpus"] = pct(entry["total"])

    # Sort by frequency (most common license first) for readable metadata.
    stats["per_license_stats"] = dict(
        sorted(per_license.items(), key=lambda kv: kv[1]["total"], reverse=True)
    )
    return stats


def _get_supported_licenses(license_file: str, data_access: DataAccess) -> list[str]:
    logger.info(f"Getting supported licenses from file {license_file}")
    licenses_list = None
    try:
        licenses_list_json, _ = data_access.get_file(license_file)
        licenses_list = json.loads(licenses_list_json.decode("utf-8"))
        logger.info(f"Read a list of {len(licenses_list)} licenses.")
    except Exception as e:
        logger.error(f"Failed to read file: {license_file} due to {e}")
    return licenses_list


class LicenseSelectTransform(AbstractTableTransform):
    """It can be used to select the rows/records of data with licenses
    matching those in the approved/deny list. It adds a new column: `license_status`
    to indicate the selected/denied licenses.

       config: dictionary of configuration data
                license_select_params: A dictionary with the following keys.
                    license_column_name - The name of the column with license, default: 'licence'.
                    allow_no_license - Allows to select rows with no license. default: False
                    licenses - A list of licenses
                    deny_licenses - if selected, the the licenses list is used as deny list, default: False
        Example:
                config = {
                    "license_select_params": {
                        "license_column_name": "license",
                        "allow_no_license": False,
                        "licenses": ["MIT", "Apache 2.0"],
                        "deny_licenses": False
                        }
                }
    """

    def __init__(self, config: dict):
        super().__init__(config)

        try:
            self.license_select = config.get(LICENSE_SELECT_PARAMS)
            self.license_column = self.license_select.get(LICENSE_COLUMN_NAME_KEY, LICENSE_COLUMN_DEFAULT)
            allow_no_license = self.license_select.get(ALLOW_NO_LICENSE_KEY, False)
            licenses = self.license_select.get(LICENSES_KEY, None)
            if not licenses or not isinstance(licenses, list):
                raise ValueError("license list not found.")
            deny = self.license_select.get(DENY_LICENSES_KEY, False)
            logger.debug(f"LICENSE_SELECT_PARAMS: {self.license_select}")
        except Exception as e:
            raise ValueError(f"Invalid Argument: cannot create LicenseSelectTransform object: {e}.")

        if not deny:
            self.transformer = AllowLicenseStatusTransformer(
                license_column=self.license_column,
                allow_no_license=allow_no_license,
                licenses=licenses,
            )
        else:
            self.transformer = DenyLicenseStatusTransformer(
                license_column=self.license_column,
                allow_no_license=allow_no_license,
                licenses=licenses,
            )

    def transform(self, table: pa.Table, file_name: str = None) -> tuple[list[pa.Table], dict]:
        """
        Transforms input tables by adding a boolean `license_status` column
        indicating whether the license is approved/denied.
        """
        TransformUtils.validate_columns(table=table, required=[self.license_column])
        new_table = self.transformer.transform(table)
        metadata = _license_stats(new_table)
        return [new_table], metadata


class LicenseSelectTransformConfiguration(TransformConfiguration):
    def __init__(self):
        super().__init__(name="license_select", transform_class=LicenseSelectTransform)

    def add_input_params(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            f"--{LICENSE_COLUMN_NAME_CLI_KEY}",
            required=False,
            type=str,
            default=LICENSE_COLUMN_DEFAULT,
            help="Name of the column holds the data to process",
        )
        parser.add_argument(
            f"--{ALLOW_NO_LICENSE_CLI_KEY}",
            required=False,
            type=lambda x: bool(str2bool(x)),
            default=False,
            help="allow entries with no associated license (default: false)",
        )
        parser.add_argument(
            f"--{LICENSES_FILE_CLI_KEY}",
            required=True,
            type=str,
            help="S3 or local path to allowed/denied licenses JSON file",
        )
        parser.add_argument(
            f"--{DENY_LICENSES_CLI_KEY}",
            type=lambda x: bool(str2bool(x)),
            required=False,
            default=False,
            help="allow all licences except those in licenses_file (default: false)",
        )
        # Create the DataAccessFactor to use CLI args
        self.daf = DataAccessFactory(CLI_PREFIX, False)
        # Add the DataAccessFactory parameters to the transform's configuration parameters.
        self.daf.add_input_params(parser)

    def apply_input_params(self, args: Namespace) -> bool:
        if not self.daf.apply_input_params(args):
            return False

        captured = CLIArgumentProvider.capture_parameters(args, CLI_PREFIX, False)
        license_column_name = captured.get(LICENSE_COLUMN_NAME_KEY)
        allow_licenses = captured.get(ALLOW_NO_LICENSE_KEY)
        deny_licenses = captured.get(DENY_LICENSES_KEY, False)
        licenses_file = captured.get(LICENSES_FILE_KEY)

        # Read licenses from allow-list or deny-list
        data_access = self.daf.create_data_access()
        licenses = _get_supported_licenses(licenses_file, data_access)

        self.params = {
            LICENSE_SELECT_PARAMS: {
                LICENSE_COLUMN_NAME_KEY: license_column_name,
                ALLOW_NO_LICENSE_KEY: allow_licenses,
                DENY_LICENSES_KEY: deny_licenses,
                LICENSES_KEY: licenses,
            }
        }
        return True
