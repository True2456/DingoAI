#!/bin/bash
# Open the drag-and-drop trajectory report viewer.

VIEWER="tools/report_viewer.html"

if [ ! -f "$VIEWER" ]; then
    echo "ERROR: $VIEWER not found."
    exit 1
fi

open "$VIEWER"
