# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM Corp. 2024.
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

import pyarrow as pa
from dpk_license_select.transform import (
    compute_license_percentages,
    _license_stats,
)


def _annotated_string_table():
    # Simulates the post-transform table (column renamed to "license" + status).
    return pa.table(
        {
            "license": ["Apache-2.0", "BAD", None, "MIT"],
            "license_status": [True, False, False, True],
        }
    )


def _annotated_list_table():
    return pa.table(
        {
            "license": pa.array(
                [
                    ["MIT"],            # kept
                    ["MIT", "GPL-3.0"],  # rejected (GPL not approved) -> both ids counted
                    [],                  # no license
                    None,                # no license
                    ["Apache-2.0"],      # kept
                ],
                type=pa.list_(pa.string()),
            ),
            "license_status": [True, False, False, False, True],
        }
    )


def test_string_column_counts():
    stats = _license_stats(_annotated_string_table())
    assert stats["total_docs"] == 4
    assert stats["docs_approved"] == 2
    assert stats["docs_rejected"] == 2
    # the None row counts as no-license AND is rejected (status False)
    assert stats["docs_no_license"] == 1
    assert stats["docs_approved"] + stats["docs_rejected"] == stats["total_docs"]
    # per-license keys are casefolded
    assert stats["lic_kept::apache-2.0"] == 1
    assert stats["lic_kept::mit"] == 1
    assert stats["lic_rejected::bad"] == 1


def test_list_column_multi_license_counts_each():
    stats = _license_stats(_annotated_list_table())
    assert stats["total_docs"] == 5
    assert stats["docs_approved"] == 2
    assert stats["docs_rejected"] == 3
    # empty list + null list are both no-license
    assert stats["docs_no_license"] == 2
    # multi-license row [MIT, GPL-3.0] was rejected -> both ids get a rejected tick
    assert stats["lic_rejected::mit"] == 1
    assert stats["lic_rejected::gpl-3.0"] == 1
    assert stats["lic_kept::mit"] == 1
    assert stats["lic_kept::apache-2.0"] == 1


def test_compute_percentages_and_reshape():
    stats = _license_stats(_annotated_list_table())
    final = compute_license_percentages(dict(stats))
    assert final["pct_docs_approved"] == 40.0  # 2/5
    assert final["pct_docs_rejected"] == 60.0  # 3/5
    assert final["pct_docs_no_license"] == 40.0  # 2/5
    # flat per-license keys removed, nested map present
    assert not any(k.startswith("lic_kept::") or k.startswith("lic_rejected::") for k in final)
    per = final["per_license_stats"]
    assert per["mit"] == {"kept": 1, "rejected": 1, "total": 2, "pct_of_corpus": 40.0}
    assert per["gpl-3.0"] == {"kept": 0, "rejected": 1, "total": 1, "pct_of_corpus": 20.0}
    assert per["apache-2.0"] == {"kept": 1, "rejected": 0, "total": 1, "pct_of_corpus": 20.0}


def test_empty_total_is_safe():
    empty = pa.table({"license": pa.array([], type=pa.string()), "license_status": pa.array([], type=pa.bool_())})
    final = compute_license_percentages(dict(_license_stats(empty)))
    assert final["total_docs"] == 0
    assert final["pct_docs_approved"] == 0.0
    assert final["per_license_stats"] == {}
