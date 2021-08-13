#!/bin/bash
USER_ID=${LOCAL_USER_ID:-9001}
echo 'Starting with username : aau and UID : -9001 host -> 1000'
useradd -s /bin/bash -u $USER_ID -o -c '' -m aau
export HOME=/home/aau
su aau bash -c 'bash'
