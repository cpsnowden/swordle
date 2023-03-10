#!/bin/bash
protobuf_location=`pip show protobuf | grep "Location" | cut -d " " -f2`
protobuf_file_location=$protobuf_location/google/protobuf/internal/

echo "Copying builder.py to $protobuf_file_location"

cp helper/builder.py $protobuf_file_location
