#!/bin/sh
set -eu
python -m featuredock.cli demo --db "${FD_DB:-runtime/events.db}"
exec python -m featuredock.cli serve --host 0.0.0.0
