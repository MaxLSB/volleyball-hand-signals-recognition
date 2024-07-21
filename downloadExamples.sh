#!/bin/bash

mkdir -p examples

url1="https://www.dropbox.com/scl/fi/xnadzviimavgunj7mlfgi/example-1.mp4?rlkey=7gax5parg1eq7bps8qrabazit&st=b4jm5x6c&dl=1"
url2="https://www.dropbox.com/scl/fi/twpo9rwxi5dofxacg0f57/example-2.mp4?rlkey=j98urfyfvf3bz6ntprjzl4wul&st=aqe0d40w&dl=1"

filename1="example-1.mp4"
filename2="example-2.mp4"

curl -L -o "examples/${filename1}" "${url1}"
curl -L -o "examples/${filename2}" "${url2}"

echo "Downloaded the videos to /examples/"
