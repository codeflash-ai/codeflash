# Copyright 2024 CodeFlash Inc. All rights reserved.
#
# Licensed under the Business Source License version 1.1.
# License source can be found in the LICENSE file.
#
# This file includes derived work covered by the following copyright and permission notices:
#
#  Copyright Python Software Foundation
#  Licensed under the Apache License, Version 2.0 (the "License").
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  The PSF License Agreement
#  https://docs.python.org/3/license.html#python-software-foundation-license-version-2
#
#

from auditwall.core import AuditWallConfig, _default_audit_wall, accept, reject


class CodeflashAuditWallConfig(AuditWallConfig):
    def __init__(self) -> None:
        super().__init__()
        self.allowed_write_paths = {".coverage", "matplotlib.rc", "codeflash"}


def handle_os_remove(event: str, args: tuple) -> None:
    filename = str(args[0])
    if any(
        pattern in filename
        for pattern in _default_audit_wall.config.allowed_write_paths
    ):
        accept(event, args)
    else:
        reject(event, args)


def check_sqlite_connect(event: str, args: tuple) -> None:
    if (
        event == "sqlite3.connect"
        and any(
            pattern in str(args[0])
            for pattern in _default_audit_wall.config.allowed_write_paths
        )
    ) or event == "sqlite3.connect/handle":
        accept(event, args)
    else:
        reject(event, args)


custom_handlers = {
    "os.remove": handle_os_remove,
    "sqlite3.connect": check_sqlite_connect,
    "sqlite3.connect/handle": check_sqlite_connect,
}


_default_audit_wall.config = CodeflashAuditWallConfig()
_default_audit_wall.config.special_handlers = custom_handlers
