#!/usr/bin/env bash
# Clone the baselines' third-party repositories at the commits the paper used (see README.md in this folder).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
fetch() {   # name url commit
    if [[ -d "$HERE/$1/.git" ]]; then
        echo "$1 exists, checking out $3"
    else
        git clone "$2" "$HERE/$1"
    fi
    git -C "$HERE/$1" checkout --quiet "$3"
    echo "$1 @ $(git -C "$HERE/$1" rev-parse HEAD)"
}
fetch TransferAttack https://github.com/Trustworthy-AI-Group/TransferAttack ae960f546a907a193931a2df12afd0f3fef2e0a5
fetch GLEAM https://github.com/LuckAlex/GLEAM 2ec57f387c0ae53d74e69d2cbaecc820589e933a
fetch DiffAttack https://github.com/WindVChen/DiffAttack e7094d04c35b2cea57b419f9e763db2605b644b7
fetch WDP https://github.com/parham1998/Facial-Privacy-Protection bd0a8e4c874b0e8bc385cbc95161033a30b8cbb7
