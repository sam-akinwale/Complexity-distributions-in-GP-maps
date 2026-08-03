#!/bin/bash

REMOTE_USER="osullivans"
REMOTE_HOST="hydra.physics.ox.ac.uk"
REMOTE_BASE="/mnt/users/osullivans/apr/teeth/plot_c"

LOCAL_DEST="./meanOPC_files"

mkdir -p "$LOCAL_DEST"

for q in Q1 Q2 Q3 Q4 Q5
do
    for m in mut0 mut1 mut2 mut3 mut4 mut5 mut6 mut7 mut8 mut9 mut10 mut11 mut12 mut13 mut14 mut15
    do
        remote_file="${REMOTE_BASE}/${q}/${m}/${q}_${m}_meanOPC.txt"

        echo "Copying ${q}/${m}..."

        scp "${REMOTE_USER}@${REMOTE_HOST}:${remote_file}" "$LOCAL_DEST/"
    done
done

echo "Done."
